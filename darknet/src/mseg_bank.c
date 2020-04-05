#include "darknet.h"
#include "image.h"
#include "mseg.h"
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
#else
#include <sys/time.h>
#endif


void train_mseg(char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
{
    
    //char *train_images = "C:/TDDownload/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt";
    //下面这个是保存权重的目录
    //char *backup_directory = "C:/TDDownload/VOCtrainval_06-Nov-2007/VOCdevkit";

    char *train_images = "C:/unet_data/train.txt";
    //下面这个是保存权重的目录
    char *backup_directory = "C:/unet_data";


    int i;
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
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

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

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
    args.classes = 20;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = MSEG_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    printf("\nzw->任一键开始训练...\n");
    _getch();//这里暂停一下
    int epoch = (*net->seen)/N;
    while(get_current_batch(*net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(*net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
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
        printf("zw->%ld, %.3f: loss:%f, %f avgl, %f rate, %lf seconds, %ld images\n", get_current_batch(*net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(*net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/(10*N) > epoch){
            epoch = *net->seen/(10*N);
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(*net, buff);
        }
        if(get_current_batch(*net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(*net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(*net, buff);

    free_network(*net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void predict_mseg(char *cfg, char *weights, char *filename)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        
		image im = load_image_color(input, 0, 0);
        //**********************************************************************************
        //image im = load_image(input, 256, 256, 1);//注意:这里输入的是灰度图像,单通道的.!
        //**********************************************************************************
        //printf("zw->c:%d w:%d h:%d adr:%d", im.c, im.w, im.h, (long)im.data);
        image sized = letterbox_image(im, net->w, net->h);//让图像在网络size的方盒子中自适应居中,空白处用0.5灰度填充
        printf("zw->in = c:%d w:%d h:%d [0]%0.2f [1]%0.2f\n", sized.c, sized.w, sized.h, sized.data[0], sized.data[1]);

        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(*net, X);
        image pred = get_network_image(*net);
        printf("zw->out= c:%d w:%d h:%d [0]%0.2f [1]%0.2f\n", pred.c, pred.w, pred.h, pred.data[0], pred.data[1]);
        
        image prmask = mask_to_rgb(pred);//这个是彩色掩码输出

        printf("Predicted: %f\n", predictions[0]);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        show_image(sized, "orig");
        show_image(prmask, "pred");
        save_image(prmask, "predictions");//zw add #flag#
        free_image(im);
        free_image(sized);
        free_image(prmask);
     
        
        wait_until_press_key_cv();//zw add #flag#
        destroy_all_windows_cv();//zw add #flag#
        
        if (filename) break;
    }
}


void demo_mseg(char *cfg, char *weights, int cam_index, const char *filename)
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


void run_mseg(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
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

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) predict_mseg(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_mseg(cfg, weights, gpus, ngpus, clear, display);
    else if(0==strcmp(argv[2], "demo")) demo_mseg(cfg, weights, cam_index, filename);
}
