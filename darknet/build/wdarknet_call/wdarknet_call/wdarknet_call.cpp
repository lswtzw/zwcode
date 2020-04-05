// wdarknet_call.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "pch.h"
#include <windows.h>
#include <iostream>
#include <processthreadsapi.h>
#include <WinUser.h>
#include <conio.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "usage: wdarknet_call [0/1/2/3.....]\n";
		printf("press any key continue.....");
		_getch();
		return 0;
	}
	//写入自个儿的tid,给通信进程
	FILE* fp = fopen("darknet_tid.bin", "rb");
	if (NULL == fp)
	{
		printf("open darknet_tid.txt failed! please run darknet preload.\n");
		printf("press any key continue.....");
		_getch();
		return 0;
	}



	//读取目标进程留下的TID
	DWORD tid=0;
	int wp = 0;
	DWORD lp = ::GetCurrentThreadId();

	try 
	{
		fread(&tid, sizeof(DWORD), 1, fp);
		fclose(fp);
		wp=atoi(argv[1]);
		
		//发送线程消息
		::PostThreadMessageA(tid, 60666, wp, lp); //实际自定义消息ID，应大于WM_USER，这里是故意测试用
		printf("tid:%d call cmd [%d] ok.\n",tid,wp);
		MSG msg;
		while (GetMessageA(&msg, NULL, 0, 65535))
		{
			if (msg.message != 60666)
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
				continue;
			}
			if(msg.lParam)printf("tid:%d replay cmd do ok.\n", tid);
			else printf("tid:%d replay cmd do failed.\n", tid);
			return 1;
		}
		return 0;
	}
	catch (...) 
	{ 
		printf("call failed.\n");
		return 0; 
	}//不处理异常


}//end main func

