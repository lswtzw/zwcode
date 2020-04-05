rem #说明: 只是预装PRELOAD模型,不识别图像,将来等wdarknet_call发来指令后,再识别一次
rem #说明: 这个要与后面的wdarknet_call.bat配合使用.
rem #      目的:节约了加载时间的快速识别

"../darknet/build/darknet/x64/darknet.exe" unet preload data.txt unet++_cfg.txt  backup/unet++_cfg.backup


