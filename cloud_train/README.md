# cloud_train — native-resolution B200 runs (2026-06)

Sequential trainer for the three pipeline models at native (~2800x4200):

| script | model | replaces |
|---|---|---|
| train_router_native.py | ResNet-18 int/ext router | pipeline/classifier_v3.pth |
| train_room_native.py | ConvNeXt-Base room classifier | room_convnext_s_v2 (89.2%) |
| train_detector_native.py | FRCNN R50 v2 cam/tripod | camtrip_frcnn_v2_cleaned (0.906) |

Data: staged at `r2:arem-training-data/cloud-train/` (originals 47.9k full-res
images + label journals + native-rescaled detector annotations).
Orchestrator: `run_all.sh` (pulls data, trains sequentially, ships ckpts+logs
to `cloud-train/runs/<stamp>/`, self-stops the pod).

Pod requirements: 1x B200 (180GB), ≥48 vCPU recommended (JPEG decode),
500GB volume, env: R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY.

On-site inference fit (probed on the 3090 at native, b1):
router 1.1GB · room ConvNeXt-B 2.9GB · detector 4.8GB — all fine.
