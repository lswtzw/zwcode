rem #说明: 这个要与preload.bat配合使用,之前必须先preload预装模型
rem #说明:如果下面的命令参数为0,则结束darknet进程.
rem #说明:如果下面的命令参数为1,则让等待中的darknet进程执行一次识别(模型已提前装入)
rem #说明:darknet进程会对data.txt文件中指定的TEST目录做单张或多张识别
rem #     目的:节约了加载时间的快速识别

"../darknet/build/darknet/x64/wdarknet_call.exe" 1

pause

