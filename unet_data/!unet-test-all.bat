rem #说明:下面用的是cpu版本的darknet_no_gpu.exe,如果你的显卡支持CUDA,可以改成darknet.exe,速度会快得多.

"../darknet/build/darknet/x64/darknet.exe" unet test data.txt unet_cfg.txt  backup/unet_cfg.backup
pause

