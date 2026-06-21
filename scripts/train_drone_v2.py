#!/usr/bin/env python3
"""Retrain the drone finish as a BLUR-PROOF tone/style model on the dngoutput
target. Two architectures (pick with --model):

  hdrnet : HDRNet/bilateral-grid. A low-res net predicts a grid of local affine
           COLOR transforms; a full-res guide slices it per-pixel; the affine is
           applied to the full-res input. out = A(x,y,luma)·in + b. The spatial
           variation is smooth (from the low-res grid) but every output pixel is
           an affine recolor of the SAME input pixel — high-freq detail is carried
           straight through, so it is STRUCTURALLY incapable of blurring.

  nafnet : the original NAFNet, but constrained: trained AND applied at native
           crop scale (no 1024-train/2048-apply delta-upscale that smeared), plus
           a gradient-preservation loss forcing output edges == input edges.

Data: <pairs_dir>/<stem>_in.jpg (developed) + _tgt.jpg (dngoutput). ~20 val holdout.
Saves best-val checkpoint to <out_ckpt>.
"""
import os, sys, glob, time, argparse, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

# ───────────────────────── data ─────────────────────────
class Pairs(torch.utils.data.Dataset):
    def __init__(self, stems, pairs_dir, crop=None, full=None):
        self.stems, self.dir, self.crop, self.full = stems, pairs_dir, crop, full
    def __len__(self): return len(self.stems)
    def _load(self, stem):
        a = cv2.imread(os.path.join(self.dir, f"{stem}_in.jpg"))
        b = cv2.imread(os.path.join(self.dir, f"{stem}_tgt.jpg"))
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        return a, b
    def __getitem__(self, i):
        a, b = self._load(self.stems[i])
        if self.full:  # fixed-size full-frame (hdrnet): letterbox-free resize to square-ish
            a = cv2.resize(a, (self.full, self.full)); b = cv2.resize(b, (self.full, self.full))
        elif self.crop:  # random crop (nafnet)
            h, w = a.shape[:2]; c = self.crop
            if h > c and w > c:
                y, x = random.randint(0, h-c), random.randint(0, w-c)
                a, b = a[y:y+c, x:x+c], b[y:y+c, x:x+c]
            else:
                a = cv2.resize(a, (c, c)); b = cv2.resize(b, (c, c))
        def t(x): return torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return t(a), t(b)

# ───────────────────────── HDRNet ─────────────────────────
class HDRNet(nn.Module):
    """Bilateral-grid color model. grid = (B, 12, D, gh, gw); 12 = 3x4 affine."""
    def __init__(self, D=8, gh=16, gw=16, lowres=256):
        super().__init__()
        self.D, self.gh, self.gw, self.lr = D, gh, gw, lowres
        def cbr(i, o, s=2): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        # low-res coefficient stream 256 -> 16
        self.coef = nn.Sequential(cbr(3, 16), cbr(16, 32), cbr(32, 64), cbr(64, 128),
                                  nn.Conv2d(128, 12 * D, 3, 1, 1))  # (B,12D,16,16)
        # pointwise guide net: full-res input -> per-pixel guide in [0,1]
        self.guide = nn.Sequential(nn.Conv2d(3, 16, 1), nn.ReLU(True), nn.Conv2d(16, 1, 1), nn.Sigmoid())
    def forward(self, x):
        B, _, H, W = x.shape
        lr = F.interpolate(x, size=(self.lr, self.lr), mode="bilinear", align_corners=False)
        grid = self.coef(lr).view(B, 12, self.D, self.gh, self.gw)  # (B,12,D,gh,gw)
        g = self.guide(x)  # (B,1,H,W) in [0,1]
        # build sampling coords for grid_sample (5D): output (B,12,1,H,W)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        gx = xs.expand(B, H, W)
        gy = ys.expand(B, H, W)
        gz = (g[:, 0] * 2 - 1)  # luma -> depth coord [-1,1]
        samp = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)  # (B,1,H,W,3) order (x=W,y=H,z=D)
        coeffs = F.grid_sample(grid, samp, mode="bilinear", align_corners=True, padding_mode="border")
        coeffs = coeffs[:, :, 0]  # (B,12,H,W)
        A = coeffs.view(B, 3, 4, H, W)  # (B, out=3, in+bias=4, H, W)
        xin = torch.cat([x, torch.ones(B, 1, H, W, device=x.device)], 1)  # (B,4,H,W)
        out = torch.einsum("boihw,bihw->bohw", A, xin)  # per-pixel affine recolor
        return out.clamp(0, 1)

def grad_loss(o, x):
    """L1 between output and INPUT spatial gradients (luma) — keep structure."""
    def lum(t): return (0.299*t[:,2]+0.587*t[:,1]+0.114*t[:,0]).unsqueeze(1)
    lo, lx = lum(o), lum(x)
    dy = lambda t: t[:, :, 1:] - t[:, :, :-1]
    dx = lambda t: t[:, :, :, 1:] - t[:, :, :, :-1]
    return (dy(lo)-dy(lx)).abs().mean() + (dx(lo)-dx(lx)).abs().mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hdrnet", "nafnet"], required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=int, default=640)      # nafnet
    ap.add_argument("--full", type=int, default=768)      # hdrnet full-frame size
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stems = sorted(s[:-7] for s in os.listdir(a.pairs) if s.endswith("_tgt.jpg"))
    random.Random(0).shuffle(stems)
    val, train = stems[:20], stems[20:]
    print(f"{len(train)} train / {len(val)} val | model={a.model} dev={dev}", flush=True)
    if a.model == "hdrnet":
        tr = Pairs(train, a.pairs, full=a.full); va = Pairs(val, a.pairs, full=a.full)
        net = HDRNet().to(dev)
    else:
        sys.path.insert(0, os.path.expanduser("~/arem-worker/pipeline"))
        from models.nafnet import NAFNet
        tr = Pairs(train, a.pairs, crop=a.crop); va = Pairs(val, a.pairs, crop=a.crop)
        net = NAFNet(in_channels=3, out_channels=3, width=32, middle_blk_num=12,
                     enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2], use_residual=True, residual_start=0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=4, drop_last=True)
    def vloss():
        net.eval(); tot = 0
        with torch.no_grad():
            for s in val:
                x, y = va[val.index(s)]; x, y = x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev)
                tot += F.l1_loss(net(x), y).item()
        net.train(); return tot/len(val)
    best = 1e9; it = 0; t0 = time.time(); di = iter(dl)
    while it < a.iters:
        try: x, y = next(di)
        except StopIteration: di = iter(dl); x, y = next(di)
        x, y = x.to(dev), y.to(dev)
        out = net(x)
        loss = F.l1_loss(out, y)
        if a.model == "nafnet":
            loss = loss + 0.5 * grad_loss(out, x)   # keep input edges -> no blur
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1
        if it % 500 == 0:
            vl = vloss()
            print(f"it {it}/{a.iters} loss {loss.item():.4f} val_L1 {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if vl < best:
                best = vl
                torch.save({"model": net.state_dict(), "arch": a.model, "val_l1": vl,
                            "version": "drone_finish_v2"}, a.out)
                print(f"  saved best {vl:.4f}", flush=True)
    print(f"DONE {a.model} best val_L1 {best:.4f}", flush=True)

if __name__ == "__main__":
    main()
