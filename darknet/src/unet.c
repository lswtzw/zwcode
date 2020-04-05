#include "darknet.h"
#include "image.h"
#include "unet.h"
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "dark_cuda.h"

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#include <processthreadsapi.h>
#include <WinUser.h>
#else
#include <sys/time.h>
#endif

LPZWDATAFILE pzwdf=NULL;
network *pgnet = NULL;
void zw_load_net( char *cfg, char *weights)
{
    if (NULL!= pgnet)return;//已装入网络就返回

    FILE* fp = fopen(cfg, "r");
    if (fp == NULL) { printf("配置文件 %s 读取有误,请检查文件是否存在,路径是否正确!", cfg); return; }//文件 不存在
    fclose(fp);

    if (NULL != weights)
    {
        FILE* fpb = fopen(weights, "rb");
        if (fpb == NULL) { printf("权重文件 %s 读取有误,请检查文件是否存在,路径是否正确!", weights); return; }//文件 不存在
        fclose(fpb);
    }

    pgnet = load_network(cfg, weights, 0);
    set_batch_network(pgnet, 1);
}//end func

void zw_load_pzwdf(char *datafile)
{
    if (NULL != pzwdf)return;

    FILE* fp = fopen(datafile, "r");
    if (fp == NULL) { printf("配置文件 %s 读取有误,请检查文件是否存在,路径是否正确!",datafile); return; }//文件 不存在
    fclose(fp); 


    pzwdf = (LPZWDATAFILE)malloc(sizeof(ZWDATAFILE));
    //zwadd
    ////////////////////////////////////////////////////////
    list *options = read_data_cfg(datafile);

    strcpy(pzwdf->test_image_path, option_find_str(options, "test_image_path", "test/"));
    strcpy(pzwdf->image_file_type_hz, option_find_str(options, "image_type_string", ".jpg"));
    pzwdf->test_start_num = option_find_int_quiet(options, "test_start_num", 10);
    pzwdf->test_end_num = option_find_int_quiet(options, "test_end_num", 99);

    strcpy(pzwdf->backup_directory, option_find_str(options, "backup", "backup/"));
    strcpy(pzwdf->train_list_file, option_find_str(options, "train_list_file", "train.list.txt"));
    strcpy(pzwdf->train_image_path, option_find_str(options, "train_image_path", "train/image/"));
    strcpy(pzwdf->train_label_image_path, option_find_str(options, "train_label_image_path", "train/label/"));


    free_list_contents_kvp(options);
    free_list(options);
}



//预装识别
void predict_unet_only_load(char *datafile, char *cfg, char *weights)
{
#ifdef WIN32

    ////////////////////////////
    if (NULL == pgnet)zw_load_net(cfg, weights);
    if (NULL == pzwdf)zw_load_pzwdf(datafile);
    ////////////////////////////
    //上面就是预装

    //写入自个儿的tid,给通信进程
    FILE* fp = fopen("darknet_tid.bin", "wb");
    DWORD tid = GetCurrentThreadId();
    fwrite(&tid, sizeof(DWORD), 1, fp);
    fclose(fp);

    int cmdjsq = 0;
    printf("(%d)->wait wdarknet_call cmd...\n",cmdjsq);
    MSG msg;
    //下面这种方式也 实测可行 而且GETMESSAGE函数是阻塞等消息 “但又不耗CPU的”！！！！。
    while (GetMessageA(&msg, NULL, 0, 65535))
    {
        if (msg.message != 60666)
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            continue;
        }

        printf("recv tid=%d send cmd %d\n", msg.lParam, msg.wParam);
        if (1 == msg.wParam)
        {
            int result=predict_unet_dir(datafile, cfg, weights);
            //自动识别测试目录下的图像文件,完成后继续等后命令,不退出
            PostThreadMessageA(msg.lParam, 60666, msg.wParam, result);//回复对方,命令成功,最后一个参数为0则表示失败.
            printf("(%d)->wait wdarknet_call cmd...\n", ++cmdjsq);
        }
        //下面两项保留给改码用户使用
        else if (2 == msg.wParam)
        {          
        }
        else if (3 == msg.wParam)
        {
        }
        else if (0 == msg.wParam)
        {
            printf("recv to over msg -> quit\n");            
            PostThreadMessageA(msg.lParam, 60666, msg.wParam, 1);//回复对方,命令成功,最后一个参数为0则表示失败.
            break;//执行这个就退出软件了
        }
    }//end while msg loop

#else
    printf("这个功能暂时只支持windows系统.\n");
#endif


   
    return;//结束函数
}//end func predict_unet_only_load



//训练函数
void train_unet(char *datafile,char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
{
    ///////////////////////////////////////////////////////////////////
    if(NULL==pzwdf)zw_load_pzwdf(datafile);//zwadd 读取data.txt配置文件
    //////////////////////////////////////////////////////////////////
    

  
    list *plist = get_paths(pzwdf->train_list_file);
    char **paths = (char **)list_to_array(plist);
    printf("Train images total nums: %d\n", plist->size);
    int N = plist->size;//样本总数

    /////////////////////////////////////////////////////////
    int i;
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("cfg file name:%s\n", base);
    printf("gpu nums:%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    image pred = get_network_image(*net);

    //下面很怪,要求输入图像与输出图像的宽度比必须是整数!!!实测如非整数倍本函数之后的代码确会死机.
    int div = net->w/pred.w;
    //printf("zwdbg->div=%d nw=%d pw=%d\n", div, net->w, pred.w);
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);
    //上面很怪,要求输入图像与输出图像的宽度比必须是整数!!!


    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);



    load_args args = {0};
   

    args.w = net->w;
    args.h = net->h;
    args.threads = 4;
    args.scale = div;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

   

    args.paths = paths;
    args.n = imgs;//每批总量
    args.m = N;//样本总数

    //试一试,从net中读取图像通道数,ok:!!!
    args.c = net->c;


    args.type = UNET_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    load_thread = load_data(args);

    printf("\nzw->任一键开始训练...\n");
    _getch();//这里暂停一下
    int epoch = (*net->seen)/N;
    while(get_current_batch(*net) < net->max_batches || net->max_batches == 0)
    {
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        //printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();
        ////////////////////////////
        //ngpus = 1;//!!!!!zw add ls
        ///////////////////////////
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(*net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(*net, train);
#endif
        if(display){
            image tr = float_to_image(net->w/div, net->h/div, 80, train.y.vals[net->batch*(net->subdivisions-1)]);
            image im = float_to_image(net->w, net->h, net->c, train.X.vals[net->batch*(net->subdivisions-1)]);
            image mask = mask_to_rgb(tr);
            image prmask = mask_to_rgb(pred);
            show_image(im, "input", 1);
            show_image(prmask, "pred", 1);
            show_image(mask, "truth", 100);
            free_image(mask);
            free_image(prmask);
        }
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("(%zd)%s->%.2f, %0.1f seconds, %ld images, loss:%0.4f, avgloss = %0.4f\n", get_current_batch(*net),cfgfile, (float)(*net->seen)/N, what_time_is_it_now()-time, *net->seen, loss, avg_loss);
        free_data(train);

        if(*net->seen/(100*N) > epoch){
            epoch = *net->seen/(100*N);
            char buff[256];
            sprintf(buff, "%s%s_%d.weights", pzwdf->backup_directory,base, epoch);
            save_weights(*net, buff);
        }
        if(get_current_batch(*net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s%s.backup", pzwdf->backup_directory,base);
            save_weights(*net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s%s.weights", pzwdf->backup_directory, base);
    save_weights(*net, buff);

    free_network(*net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);

}






//批量处理一个目录下的所有图像,成功返回1,失败返回0
int predict_unet_dir(char *datafile, char *cfg, char *weights)
{
    if(NULL == pgnet)zw_load_net(cfg, weights);
    if (NULL == pzwdf)zw_load_pzwdf(datafile);

    srand(2222222);

    clock_t st_time;
    char buff[256];
    char *input = buff;

    int erjsq = 0;
    for (int n = pzwdf->test_start_num; n <= pzwdf->test_end_num; n++)
    {
        char outfnbuf[256];
        sprintf(input, "%s%d%s", pzwdf->test_image_path,n, pzwdf->image_file_type_hz);
        sprintf(outfnbuf, "%s%d_pred", pzwdf->test_image_path, n);
        //**********************************************************************************
        FILE* fp = fopen(input, "rb");
        if (fp == NULL) { erjsq++; if (erjsq > 100) { return 0; } else { continue; } }//文件 不存在
        fclose(fp); erjsq = 0;


        st_time = clock();
        image im = { 0 };
        image sized = { 0 };
        image pred = { 0 };
        image prmask = { 0 };

        if (pgnet->c <= 1)
            im = load_image_gray(input, 0, 0);//注意:这里输入的是灰度图像,单通道的.!
        else
            im = load_image_color(input, 0, 0);
        //**********************************************************************************


        //printf("zw->c:%d w:%d h:%d adr:%d", im.c, im.w, im.h, (long)im.data);
        sized = letterbox_image(im, pgnet->w, pgnet->h);//让图像在网络size的方盒子中自适应居中,空白处用0.5灰度填充
        //printf("zw->in = c:%d w:%d h:%d [0]%0.2f [1]%0.2f [2]%0.2f\n", sized.c, sized.w, sized.h, sized.data[0], sized.data[1], sized.data[2]);

        float *X = sized.data;

        float *predictions = network_predict(*pgnet, X);
        pred = get_network_image(*pgnet);
        //printf("zw->out= c:%d w:%d h:%d [0]%0.2f [1]%0.2f [2]%0.2f\n", pred.c, pred.w, pred.h, pred.data[0], pred.data[1], pred.data[2]);

        //////////////////////////////////////////////////
        if (pgnet->c <= 1)//单通道 
            prmask = pred;
        else
            prmask = mask_to_rgb(pred);//这个是彩色掩码输出
            ///////////////////////////////////////////////////
        printf("pred image file: %s over. used:%0.3f seconds\n", input, sec(clock() - st_time));
        save_image(prmask, outfnbuf);//zw add #flag#

       //注意:pred的图像是不能free的,它是指针指向net的实体
        free_image(im);       
        free_image(sized);
        if (pgnet->c > 1)free_image(prmask);
    }//end for


    return 1;
}//end func


void predict_unet(char *datafile,char *cfg, char *weights, char *filename)
{
    if (NULL == filename)//不是识别一个文件 ,而是一个目录下的多文件
    {
        return predict_unet_dir(datafile, cfg, weights);       
    }

    //下面载入模型和配置
    if (NULL == pgnet)zw_load_net(cfg, weights);
    if (NULL == pzwdf)zw_load_pzwdf(datafile);



    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;


    strncpy(input, filename, 256);
    //**********************************************************************************
    //printf("cfgfile:c=%d\n", net->c);
    time = clock();
    image im = { 0 };
    if(pgnet->c<=1)
    im= load_image_gray(input, 0, 0);//注意:这里输入的是灰度图像,单通道的.!
    else
    im = load_image_color(input, 0, 0);
    //**********************************************************************************

    if (im.w == 10 && im.h == 10)
    {

        printf("文件不存在!或输入的路径不完整!\n");

        _getch();
        return;
    }


    //printf("zw->c:%d w:%d h:%d adr:%d", im.c, im.w, im.h, (long)im.data);
    image sized = letterbox_image(im, pgnet->w, pgnet->h);//让图像在网络size的方盒子中自适应居中,空白处用0.5灰度填充
    printf("zw->in = c:%d w:%d h:%d [0]%0.2f [1]%0.2f [2]%0.2f\n", sized.c, sized.w, sized.h, sized.data[0], sized.data[1], sized.data[2]);

    float *X = sized.data;
  
    float *predictions = network_predict(*pgnet, X);
    image pred = get_network_image(*pgnet);
    printf("zw->out= c:%d w:%d h:%d [0]%0.2f [1]%0.2f [2]%0.2f\n", pred.c, pred.w, pred.h, pred.data[0], pred.data[1], pred.data[2]);

    //////////////////////////////////////////////////
    image prmask = { 0 };
    if (pgnet->c <= 1)//单通道 
        prmask = pred;
    else
        prmask = mask_to_rgb(pred);//这个是彩色掩码输出
        ///////////////////////////////////////////////////


    printf("Predicted: %f\n", predictions[0]);
    printf("%s: Predicted in %0.3f seconds.\n", input, sec(clock() - time));
    show_image(sized, "orig");
    show_image(prmask, "pred");
    save_image(prmask, "predictions");//zw add #flag#
    free_image(im);
    free_image(sized);
    free_image(prmask);


    wait_until_press_key_cv();//zw add #flag#
    destroy_all_windows_cv();//zw add #flag#


}//end func


//注意:下面这个函数我并没有测试过,可能存在很多问题
void demo_unet(char *data,char *cfg, char *weights,char *filename,int cam_index)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    srand(2222222);
    cap_cv * cap;

    if (filename) {
        cap = get_capture_video_stream(filename);
    }
    else {
        cap = get_capture_webcam(cam_index);
    }


    if(!cap) error("Couldn't connect to webcam.\n");
    create_window_cv("mseg", 0, 512, 512);

    float fps = 0;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        //image in = get_image_from_stream(cap);
        image in = get_image_from_stream_cpp(cap);
        image in_s = letterbox_image(in, net->w, net->h);
        show_image(in, "mseg");

        network_predict(*net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        image pred = get_network_image(*net);
        image prmask = mask_to_rgb(pred);
        show_image(prmask, "Segmenter", 10);
        
        free_image(in_s);
        free_image(in);
        free_image(prmask);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void run_unet(int argc, char **argv)
{
    if(argc < 5){
        fprintf(stderr, "usage: darknet unet <train/test/preload> <data filename> <cfg filename> [weights filename] [image file name]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int cam_index = find_int_arg(argc, argv, "-c", 0);//这个是取参数的设置值,和下两个不同
    int clear = find_arg(argc, argv, "-clear");//非0,表示命令行上加了这个参数
    int display = find_arg(argc, argv, "-display");//非0,表示命令行上加了这个参数

    //char *cfg = argv[3];
    //char *weights = (argc > 4) ? argv[4] : 0;
    //char *filename = (argc > 5) ? argv[5]: 0;

    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;

    if(0==strcmp(argv[2], "test")) predict_unet(data,cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_unet(data,cfg, weights, gpus, ngpus, clear, display);
    //else if(0==strcmp(argv[2], "demo")) demo_unet(data,cfg, weights,filename,cam_index);
    else if (0 == strcmp(argv[2], "preload"))predict_unet_only_load(data, cfg, weights);
}


