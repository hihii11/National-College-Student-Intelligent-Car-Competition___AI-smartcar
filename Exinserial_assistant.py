import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import threading
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter.filedialog
import pygal
import pyqtgraph as pg

top=tk.Tk()#创建消息循环
top.title('Exin_port_assistant-1.0')
top.geometry("500x420")
ser=''
open_flag=0
show_flag=0
data_rec=[]
def find_port():#该函数用于查询可用串口
    screen1["text"]="正在查询串口!请稍后......"
    top.update()
    port_list = list(serial.tools.list_ports.comports())#将串口信息写入port_list
    if len(port_list) == 0:
        screen1["text"] = "无可用串口!"
        top.update()
    else:
        screen1["text"]="可用串口:"
        for i in range(0,len(port_list)):
            screen1["text"]=screen1["text"]+str(port_list[i])+'\n'#打印串口信息
        top.update()
    return port_list
def set_port(port,baunds):#设置串口参数，打开串口
    global ser
    ser=serial.Serial(port,baunds, timeout=None)  # 使用USB连接串行口
def txt_clr():#清空显示窗口
    E1.delete('1.0','end')
def thread_recv():#串口接收数据线程
    global ser,tmp,dat_num
    while True:
        read = ser.read()
        if read !='':
            tmp=bytes(read).hex()
            data_rec.append(tmp)
            data_rec.append(' ')
def thread_show():#接收数据显示线程，由于显示较慢，所以设定为停止接收后显示
    global show_flag
    count=0
    data=data_rec
    for i in range(len(data)):
        show_flag = 1
        top.update()
        E1.insert('end', str(data[i]))#滚动文本框插入数据
        E1.see('end')#跟踪文本框尾部
        count=count+1
        if(count==200):#显示数200时情况文本框
            count=0
            txt_clr()
    show_flag = 0
def txt_ctl():#将数据写入txt文本。txt文件名由file_name1指定
    file = open(file_name1.get(), 'w')
    for num in data_rec:
        file.write(str(num).upper())
    file.close()
def port_ctl():#控制串口打开/关闭
    global open_flag,ser,num_c,data_rec,show_flag
    open_flag=~open_flag
    try:
        if(open_flag!=0):
            b1["text"]="关闭串口"
            set_port(str(Bin0.get()),str(Bin1.get()))
            screen1["text"] = str(ser.port)+"打开"+"\n"\
                                +"波特率:"+str(ser.baudrate)
            recv_data = threading.Thread(target=thread_recv)#开启数据接收线程
            recv_data.start()#开启数据接收线程
        else:
            b1["text"] = "打开串口"
            ser.close()
            screen1["text"] = str(ser.port) + "关闭" + "\n"\
            + "接收数:" + str(len(data_rec))
            txt_ctl()
            if(show_flag==0):
                print('Y')
                txt_clr()
                #show_data = threading.Thread(target=thread_show)
               # show_data.start()
            eb3.set(eb1.get())
        print(open_flag)
    except:
            screen1["text"] = Bin0.get() +"连接失败"
            open_flag = ~open_flag
            b1["text"] = "打开串口"
#移出文件，这里不需要带上文件类型只需要打文件名字如要删除data.txt,则文本框输入data
#若要删除所有数据，则需在文本框打入all
def file_move():
    name=file_name2.get()#获得文本框数值
    screen1["text"] = '删除:'
    if(name=='all'):#删除所有数据的情况
        path='.'
        files = os.listdir(path)
        for file in files:
            if (file.split(".")[-1] == 'npy'):
                if (file[0:9] == 'origin_ad'):
                    os.remove(file)
                    screen1["text"]= screen1["text"]+file+'\n'
                elif (file[0:10] == 'origin_pwm'):
                    os.remove(file)
                    screen1["text"] = screen1["text"] + file + '\n'
                elif (file[0:2] == 'ad'):
                    os.remove(file)
                    screen1["text"] = screen1["text"] + file + '\n'
                elif (file[0:3] == 'pwm'):
                    os.remove(file)
                    screen1["text"] = screen1["text"] + file + '\n'
            if (file.split(".")[-1] == 'txt'):
                if (file[0:8] != 'smartcar'):
                    os.remove(file)
                    screen1["text"] = screen1["text"] + file + '\n'
    else:#删除指定数据的请况，这里删除npy与txt文件
        path = '.'
        files = os.listdir(path)
        len1=len(name)
        print(len1)
        for file in files:
            len2 = len(file)
            print(file[len2-1-4-len1+1:len2-1-4])
            if (file.split(".")[-1] == 'txt'):
                if(file[len2-1-4-len1+1:len2-4]==name):
                    os.remove(file)
                    screen1["text"] = screen1["text"] + file + '\n'
            if (file.split(".")[-1] == 'npy'):
                if(file[len2-1-4-len1+1:len2-4]==name):
                    os.remove(file)
                    screen1["text"] = screen1["text"] + file + '\n'
def data_deal():#数据处理
    screen1["text"]='正在处理数据:'
    top.update()
    split()
    screen1["text"] = '正在处理数据:▋▋▋'+'\n'+'生成特征(ad)标签(pwm)'
    top.update()
    convert()
    screen1["text"] = '正在处理数据:▋▋▋▋▋▋'+'\n'+'生成训练数据'
    top.update()
    con()
    screen1["text"] = '正在处理数据:▋▋▋▋▋▋▋▋▋'+'\n'+'数据处理完成'+'\n'+\
                        '训练集文件:'+'ad_train_dat.npy(特征)'+'\n'+\
                                'pwm_train_dat.npy(标签)'+'\n'+ \
                      '测试集文件:' + 'ad_test_dat.npy(特征)' + '\n' + \
                                  'pwm_test_dat.npy(标签)'
    top.update()
#寻找噪点
def find_label_noise_pos(label):
    max = 0
    for i in range(label.size - 1):
        if (max < abs(label[i + 1] - label[i])):
            max = abs(label[i + 1] - label[i])
            pos = i + 1

    return pos, max
#ad值的一阶滞后滤波
def ad_data_denoise(ad_array):
    rate = 0.8
    ad_new = []
    idx = 0
    oldvalue = ad_array[0]
    for i in range(ad_array.size):
        new = ad_array[i] * rate + (1 - rate) * oldvalue
        oldvalue = new
        ad_new.append(new)

    rt_array = np.array(ad_new)
    return rt_array
def ad_array_extension(ad_array, batch_size):
    ad_list = []#创建一个ad空链表用于存放处理后的ad值
    list_len = ad_array.shape[0]
    for i in range(list_len):
        if (i > batch_size):
            ad_list.append([ad_array[i - batch_size:i]])
        else:
            ad_list.append([ad_array[0:batch_size]])
    ad_new = np.array(ad_list)#再次转换为numpy类型

    return ad_new
#情况接收数据缓冲区，在更改存放的数据目录时需要调用
def dat_clr():
    global data_rec
    data_rec=[]
    screen1["text"]='缓冲区已清空'
#对数据进行划分
def split():
    num = 0
    filename = file_name3.get()
    fp1 = open(filename, 'r+')
    screen1["text"] = '正在处理数据:▋▋'+'\n'+'打开文件'+filename
    top.update()
    str2 = fp1.read()
    str1=''
    str3=''
    str4=''
    for i in str2:
        if i == ' ':  # 找到空格
            num += 1  # 数量+1
            str3 += str1 + i
            if str1 == "5A":  # 找到固定数据
                if num < 11:  # 数据不够 继续统计
                    pass
                elif num == 11:  # 找到一组数据
                    str3 += "\r"  # 插入换行
                    str4 += str3  # 保存获得的字符串
                    num = 0  # 清空统计
                    str3 = ""
                else:  # 固定数据出现丢失 导致一组数据过多 丢弃
                    num = 0  # 清空统计
                    str3 = ""

            str1 = ""  # 清空
        else:
            str1 += i
    fp1.close()
    fp2 = open("split_" + filename, 'w+')
    fp2.write(str4)
    fp2.close()
     #screen["text"] = 'Welcome to AI Team!!!\n The file split SUCCESSFULLY!!ლ(╹◡╹ლ)'
#将当前目录下所有的npy类型数据文件集合，然后生成数据集和测试集
#注意:测试集的大小为总数据的20％
def con():
        path = "."
        ad_array = []
        pwm_array = []
        files = os.listdir(path)
        for file in files:
                if (file.split(".")[-1] == 'npy'):
                        if (file[0:9] == 'origin_ad'):
                                ad = np.load(file)
                                if (len(ad_array) == 0):
                                        ad_array = ad
                                else:
                                        ad_array = np.concatenate((ad_array, ad))
                        elif (file[0:10] == 'origin_pwm'):
                                pwm = np.load(file)
                                if (len(pwm_array) == 0):
                                        pwm_array = pwm
                                else:
                                        pwm_array = np.concatenate((pwm_array, pwm))
        screen1["text"] = '正在处理数据:▋▋▋▋▋▋▋'+'\n'+'读取数据文件'
        top.update()
        ad_dat = ad_array
        pwm_dat = pwm_array
        out_len = int(len(ad_dat) / 100) * 100
        test_ad_npy = ad_dat[0:out_len]
        label_pwm_npy = pwm_dat[0:out_len]
        rest_train = ad_dat[out_len:len(ad_dat)]
        rest_label = pwm_dat[out_len:len(ad_dat)]
        screen1["text"] = '正在处理数据:▋▋▋▋▋▋▋▋'+'\n'+'生成训练集/测试集 比例为 4:1'
        top.update()
        print(test_ad_npy.size, len(test_ad_npy))
        print(test_ad_npy.shape)

        train_data, test_data, train_label, test_label = train_test_split(test_ad_npy, label_pwm_npy, test_size=0.2,
                                                                          random_state=0)
        train_data = np.concatenate((train_data, rest_train))
        train_label = np.concatenate((train_label, rest_label))

        print('ad train data shape:%f~%f' % (max(train_data.flatten()), min(train_data.flatten())))
        print('pwm train data shape:%f~%f' % (max(train_label), min(train_label)))

        np.save('./ad_train_dat.npy', train_data)
        np.save('./ad_test_dat.npy', test_data)

        print('ad test data shape:%f~%f' % (max(test_data.flatten()), min(test_data.flatten())))
        print('pwm test data shape:%f~%f' % (max(test_label), min(test_label)))
        np.save('./pwm_train_label.npy', train_label)
        np.save('./pwm_test_label.npy', test_label)

        print('Generate ad_train_dat.npy,ad_test_dat.npy,pwm_train_label.npy,pwm_test_label.npy')
        #screen["text"]=('Welcome to AI Team!!!\n The file concatenate SUCCESSFULLY!!ლ(╹◡╹ლ)')
#将pwm数据与电感ad值分为标签与特征，类型为npy
def convert():
        file_name='split_'+file_name3.get()
        dat_str = np.loadtxt(file_name, dtype=np.str, delimiter=' ', unpack=False)
        # 转换为一维数组 [1 2 3 4 5]
        dat_str = dat_str.reshape(dat_str.size)
        # 定义一个list
        dat_int = []
        # 遍历字符串读取每个数据
        batch_size=1
        for a in dat_str:
                if (len(a) > 0):
                        hex_str = '0x' + a  # 将数据转换为 0xXX
                        dat_int.append(int(hex_str, 16))  # 将数据转换为十进制并添加到dat_int列表
        # 将列表转换为数组便于使用
        dat_int = np.array(dat_int)
        # 计算有多少组完整数据
        lenth = int(int(len(dat_int) / 11) * 11)
        # 这是一步多余的操作
        dat_int = dat_int[0:lenth]
        # 将数据变形为矩阵 也可以说是二维数组
        dat_int = dat_int.reshape(int(dat_int.size / 11), 11)
        # 定义一个list
        list = []
        screen1["text"]='正在处理数据:▋▋▋▋'+'\n'+'遍历矩阵'
        top.update()
        # 遍历dat_int矩阵
        for a in dat_int:
                if (a[10] != 90):
                        screen1["text"] = '正在处理数据:▋▋▋▋' + '\n' + '数据错误!'
                        print(a)
                        raise ValueError('Error data')
                        break
                # 计算时间
                time = (a[0] << 8) + a[1]
                # print('%d,%d,%d,%d' % (a[0], a[1], a[2],time))
                # 获取电感与PWM的数值
                l = a[2]
                ml = a[3]
                m = a[4]
                mr = a[5]
                r = a[6]
                fl = a[7]
                fr = a[8]
                pwm = a[9]
                # 将获取到的数值添加到list
                list.append([l, ml, m, mr, r, fl, fr, pwm])
        # 将list转换为数组 并将数据类型转换为有符号8位
        list = (np.array(list)).astype(np.int8)
        # 通过切片操作 获得ad数据
        ad_dat = list[:, [0, 1, 2, 3, 4, 5, 6]]
        # 通过切片操作 获得pwm数据
        pwm_dat = list[:, 7]
        # 新建一个ad数据
        dbg_ad = ad_dat
        # 通过切片操作读取每个通道的数值
        a_l = ad_data_denoise((list[:, 0]))
        a_ml = ad_data_denoise((list[:, 1]))
        a_m = ad_data_denoise((list[:, 2]))
        a_mr = ad_data_denoise((list[:, 3]))
        a_r = ad_data_denoise((list[:, 4]))
        a_fl = ad_data_denoise((list[:, 5]))
        a_fr = ad_data_denoise((list[:, 6]))
        pwm_dat = list[:, 7]
        # 转换数据类型为8位有符号数据 并通过ad_data_denoise函数滤波
        pwm_dat = ad_data_denoise(pwm_dat.astype('int8'))
        screen1["text"] = '正在处理数据:▋▋▋▋▋'+'\n'+'连接切片数据'
        top.update()
        # 将切片对象按列连接
        ad_dat = np.c_[a_l, a_ml, a_m, a_mr, a_r, a_fl, a_fr]
        # 转换为有符号 不明白为啥要这样做 前面不是转换好了？
        ad_dat = ad_dat.astype('int8')
        # 扩展AD数据  实际上就是复制数据
        ad_dat = ad_array_extension(ad_dat, batch_size)
        # 将数据转换为三维数组 最外层与有多少组ad数据一致 中间层有参数指定 最内层为7个电感数据
        ad_dat = ad_dat.reshape(ad_dat.shape[0], batch_size, 7)
        origin_ad = ad_dat.astype('int8')
        origin_pwm = pwm_dat.astype('int8')
        # print(args.file)
        # 获取文件名称以及格式
        filename = file_name.split("\\")[-1]
        # print(filename)
        # 获取文件名
        filename = filename.split(".")[0]
        # print(filename)
        # 准备输出文件的名称
        pwm_fname = 'origin_pwm_' + filename + '.npy'
        ad_fname = 'origin_ad_' + filename + '.npy'
        np.save(pwm_fname, origin_pwm)
        np.save(ad_fname, origin_ad)
        print("Generate ad.npy shape:", ad_dat.shape)
        print("Generate pwm.npy shape:", pwm_dat.shape)
        print('ad data shape:%f~%f' % (max(origin_ad.flatten()), min(origin_ad.flatten())))
        print('pwm data shape:%f~%f' % (max(origin_pwm), min(origin_pwm)))
        print('Generate %s,%s' % (pwm_fname, ad_fname))
              #  screen["text"]='Welcome to AI Team!!!\n The file converttest SUCCESSFULLY!!ლ(╹◡╹ლ)'
#计算数据均值，标准差,均方差
def mean_var_std(arr):
    arr_mean = np.mean(arr)
    arr_var = np.var(arr)
    arr_std = np.std(arr, ddof=1)
    return arr_mean,arr_var,arr_std
#画图
def save_add_init(path):
    isExists = os.path.exists(path)
    if isExists == 0:
        os.makedirs(path)
def draw():
    save_add_init(file_name4.get() + '折线图')
    #将file_name4指定的数据文件打入data数组
    data = np.loadtxt(file_name4.get(), dtype=np.str, delimiter=' ', unpack=False)
    #计算显示数
    x = int(x_table_end.get()) - int((x_table_start.get()))
    #重构data数组
    data = data.reshape(data.size)
    dat_int = []
    for a in data:
        if (len(a) > 0):
            hex_str = '0x' + a  # 将数据转换为 0xXX
            dat_int.append(int(hex_str, 16))#转换为16进制
    data = np.array(dat_int)#将数组转换为npy类型
    # 将数据变形为矩阵 也可以说是二维数组
    data = data.reshape(int(data.size / 11), 11)
    data = data.astype(Bin_type.get())
    x1 = data[int(x_table_start.get()):int(x_table_end.get()), 2]#电感1
    x2 = data[int(x_table_start.get()):int(x_table_end.get()), 3]#电感2
    x3 = data[int(x_table_start.get()):int(x_table_end.get()), 4]#电感3
    x4 = data[int(x_table_start.get()):int(x_table_end.get()), 5]#电感4
    x5 = data[int(x_table_start.get()):int(x_table_end.get()), 6]#电感5
    x6 = data[int(x_table_start.get()):int(x_table_end.get()), 7]#电感6
    x7 = data[int(x_table_start.get()):int(x_table_end.get()), 8]#电感7
    x8 = data[int(x_table_start.get()):int(x_table_end.get()), 9]#pwm
    '''
    top.update()
    view = pygal.Line()
    view.add('L1',x1)
    view.render_to_file(file_name4.get() + '折线图'+'\\'+'L1.svg')
    '''
    x=np.arange(0.0,len(x1),1.0)
    screen1["text"] = '正在绘图:'  + '(1/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce1')
    plt.plot(x, x1)
    plt.savefig(file_name4.get() + '折线图'+'\\'+'L1.png')
    plt.close()
    top.update()
    screen1["text"] = '正在绘图:▋' + '(2/9)'
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce2')
    plt.plot(x, x2)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L2.png')
    plt.close()
    top.update()
    screen1["text"] = '正在绘图:▋▋' + '(3/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce3')
    plt.plot(x, x3)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L3.png')
    plt.close()
    screen1["text"] = '正在绘图:▋▋▋' + '(4/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce4')
    plt.plot(x, x4)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L4.png')
    plt.close()
    screen1["text"] = '正在绘图:▋▋▋▋' + '(5/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce5')
    plt.plot(x, x5)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L5.png')
    plt.close()
    screen1["text"] = '正在绘图:▋▋▋▋▋' + '(6/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce6')
    plt.plot(x, x6)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L6.png')
    plt.close()
    screen1["text"] = '正在绘图:▋▋▋▋▋▋' + '(7/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('induce7')
    plt.plot(x, x7)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L7.png')
    plt.close()
    screen1["text"] = '正在绘图:▋▋▋▋▋▋▋' + '(8/9)'
    top.update()
    plt.figure(1)
    plt.subplot(211)
    plt.title('PWM_OUT')
    plt.plot(x, x8)
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'PWM.png')
    plt.close()
    screen1["text"] = '正在绘图:▋▋▋▋▋▋▋▋' + '(9/9)'
    top.update()
    '''
    w = pg.GraphicsWindow()
    plot = w.addPlot(title='电感ad值-L1:绿 L2:红 L3:黄 L4:天蓝 L5:粉 L6:白 L7:银 ')
    plot.plot(x1,pen='g',name='L1')
    plot.plot(x2,pen='r',name='L2')
    plot.plot(x3, pen='y', name='L3')
    plot.plot(x4, pen='c', name='L4')
    plot.plot(x5, pen='m', name='L5')
    plot.plot(x6, pen='w', name='L6')
    plot.plot(x7, pen='d', name='L7')
    #ex = pyqtgraph.exporters.ImageExporter(w.scene())
    #ex.export(fileName=file_name4.get() + '折线图'+'\\'+'L1..L7.png')
    pg.QtGui.QGuiApplication.exec_()
    '''
    '''
    view = pygal.Line()
    view.add('L1', x1)
    view.add('L2',x2)
    view.render_to_file(file_name4.get() + '折线图' + '\\' + 'L1....L7.svg')
    '''
    plt.title('induce')
    plt.figure(1)
    plt.plot(x, x1,label='L1')
    plt.plot(x, x2,label='L2')
    plt.plot(x, x3,label='L3')
    plt.plot(x, x4,label='L4')
    plt.plot(x, x5,label='L5')
    plt.plot(x, x6,label='L6')
    plt.plot(x, x7,label='L7')
    plt.legend()
    plt.savefig(file_name4.get() + '折线图' + '\\' + 'L1...L7.png')
    screen1["text"] = "PWM均值\方差\标准差:" + str(mean_var_std(x8)[0]) + ' \\ ' + str(mean_var_std(x8)[1]) + ' \\ ' + str(
        mean_var_std(x8)[2]) + '\n' \
                      + "L1均值\方差\标准差:" + str(mean_var_std(x1)[0]) + ' \\ ' + str(mean_var_std(x1)[1]) + ' \\ ' + str(
        mean_var_std(x1)[2]) + '\n' \
                      + "L2均值\方差\标准差:" + str(mean_var_std(x2)[0]) + ' \\ ' + str(mean_var_std(x2)[1]) + ' \\ ' + str(
        mean_var_std(x2)[2]) + '\n' \
                      + "L3均值\方差\标准差:" + str(mean_var_std(x3)[0]) + ' \\ ' + str(mean_var_std(x3)[1]) + ' \\ ' + str(
        mean_var_std(x3)[2]) + '\n' \
                      + "L4均值\方差\标准差:" + str(mean_var_std(x4)[0]) + ' \\ ' + str(mean_var_std(x4)[1]) + ' \\ ' + str(
        mean_var_std(x4)[2]) + '\n' \
                      + "L5均值\方差\标准差:" + str(mean_var_std(x5)[0]) + ' \\ ' + str(mean_var_std(x5)[1]) + ' \\ ' + str(
        mean_var_std(x5)[2]) + '\n' \
                      + "L6均值\方差\标准差:" + str(mean_var_std(x6)[0]) + ' \\ ' + str(mean_var_std(x6)[1]) + ' \\ ' + str(
        mean_var_std(x6)[2]) + '\n' \
                      + "L7均值\方差\标准差:" + str(mean_var_std(x7)[0]) + ' \\ ' + str(mean_var_std(x7)[1]) + ' \\ ' + str(
        mean_var_std(x7)[2]) + '\n'
    plt.show()
    os.startfile(file_name4.get() + '折线图')
#示波器
osci_flag=0
shiftt=[]
data=[]
osci_x=[]
osci_thread_alive = 0

app = pg.mkQApp()
win = pg.GraphicsWindow()
win.setWindowTitle(u'pyqtgraph plot demo')
win.resize(600, 400)
p = win.addPlot()
p.showGrid(x=True, y=True)
p.setLabel(axis='left', text=u'value ')
p.setLabel(axis='bottom', text=u'times / n')
p.setTitle('Oscilloscope')
p.addLegend()
curve1 = p.plot(pen='r', name='port_data')

def osci_open():
    global  osci_flag,shiftt,data,osci_x,win,p1,curve1
    global osci_thread_alive
    osci_flag = ~osci_flag
    #pg.QtGui.QGuiApplication.exec_()
    if (osci_flag != 0):
        bottom4["text"] = "关闭示波器"
        osci_x=[]
        data=[]
        shiftt=[]
        if(osci_thread_alive==0):
            osci_thread = threading.Thread(target=oscilloscope)  # 开启数据接收线程
            osci_thread.setDaemon(True)
            osci_thread.start()  # 开启数据接收线程
            osci_thread_alive=1
    else:
        bottom4["text"] = "打开示波器"
    print(osci_flag)
def oscilloscope():
    global osci_flag,shiftt,data,osci_x,win,p1,curve1,app
    app.exec_()
    pre_data=[]
    while True:
        #try:
        if(osci_flag!=0):
            once_close = 0
            lenth=int(Bin_range.get())#示波器显示长度
            #osc_data=[0 for _ in range(lenth)]#示波器缓冲区
            if len(data_rec)<1000*11*2:
                lenth=len(data_rec)
            #从数据中取出1000*11*2组数据
            shiftt=data_rec[len(data_rec)-1-1000*11*2:len(data_rec)-1]
            file = open('oscilloscope.txt', 'w')
            for num in shiftt:
                file.write(str(num).upper())
            file.close()
            num = 0
            filename = 'oscilloscope.txt'
            fp1 = open(filename, 'r+')
            str2 = fp1.read()
            str1 = ''
            str3 = ''
            str4 = ''
            for i in str2:
                if i == ' ':  # 找到空格
                    num += 1  # 数量+1
                    str3 += str1 + i
                    if str1 == "5A":  # 找到固定数据
                        if num < 11:  # 数据不够 继续统计
                            pass
                        elif num == 11:  # 找到一组数据
                            str3 += "\r"  # 插入换行
                            str4 += str3  # 保存获得的字符串
                            num = 0  # 清空统计
                            str3 = ""
                        else:  # 固定数据出现丢失 导致一组数据过多 丢弃
                            num = 0  # 清空统计
                            str3 = ""

                    str1 = ""  # 清空
                else:
                    str1 += i
            fp1.close()
            fp2 = open("split_" + 'oscilloscope.txt', 'w+')
            fp2.write(str4)
            fp2.close()
            data = np.loadtxt("split_" + 'oscilloscope.txt', dtype=np.str, delimiter=' ', unpack=False)
            data = data.reshape(data.size)
            dat_int = []
            for a in data:
                if (len(a) > 0):
                    hex_str = '0x' + a  # 将数据转换为 0xXX
                    dat_int.append(int(hex_str, 16))  # 转换为16进制
            data = np.array(dat_int)  # 将数组转换为npy类型
            # 将数据变形为矩阵 也可以说是二维数组
            #print(data.size/11,data.size%11)
            try:
                #screen1["text"] = '帧数正常'
                #top.update()
                data = data.reshape((int(data.size / 11), 11))
            except:
                #screen1["text"]='帧数错误'
                #top.update()
                data = data[:data.size - data.size % 11]
                data = data.reshape((int(data.size / 11), 11))
                #data = pre_data
            #pre_data=data
            data = data.astype('int8')
            if(Bin_channel.get()=='L1(电感1)'):
                osci_x = data[len(data)-1-lenth:len(data)-1, 2]  # 电感1
            if (Bin_channel.get() == 'L2(电感2)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 3]  # 电感2
            if (Bin_channel.get() == 'L3(电感3)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 4]  # 电感3
            if (Bin_channel.get() == 'L4(电感4)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 5]  # 电感4
            if (Bin_channel.get() == 'L5(电感5)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 6]  # 电感5
            if (Bin_channel.get() == 'L6(电感6)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 7]  # 电感6
            if (Bin_channel.get() == 'L7(电感7)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 8]  # 电感7
            if (Bin_channel.get() == 'pwm(舵机打角)'):
                osci_x = data[len(data) - 1 - lenth:len(data) - 1, 9]  # pwm
            curve1.setData(osci_x)
        #except:
         #       print('drwa_wrong')

#------------------------------------------------------------------
#处理信息显示屏幕GUI
screen1=tk.Label(top,text='hello!!',font=('Arial',8),width=45,
            borderwidth=5,
         relief="groove",
         height=9)
screen1.grid(row=0,column=3,columnspan=40,rowspan=5)
#------------------------------------------------------------------
#有关串口GUI设计
b0=tk.Button(top,text='查询可用串口',font=('Arial',10),width=10,height=1,command= find_port,borderwidth=5,relief="groove")
b0.grid(row=3,column=0)
b1=tk.Button(top,text='打开串口',font=('Arial',10),width=10,height=1,command= port_ctl,borderwidth=5,relief="groove")
b1.grid(row=3,column=1)
tk.Label(top,text='当前串口:',font=('Arial',10),width=12,
         height=1,justify=tk.LEFT,
         borderwidth=1,
         relief="groove").grid(row=0,column=0)

tk.Label(top,text='波特率:',font=('Arial',10),width=12,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=1,column=0)
def go3(*args):
    print(Bin1.get())
e14 = tk.StringVar(value='115200')
Bin1 = ttk.Combobox(textvariable=e14, width=12)
Bin1["values"] = ('230400', '115200', '38400', '28800', '9600', '4800', '2400', '1200', '600')
Bin1.current(0)
Bin1.bind("<<Binselect>>", go3)
Bin1.grid(row=1, column=1)
def go2(*args):
    print(Bin0.get())
e0 = tk.StringVar(value='COM1')
Bin0 = ttk.Combobox(textvariable=e0, width=12)
Bin0["values"] = (
'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10', 'COM11', 'COM12', 'COM13', 'COM14',
'COM15')
Bin0.current(0)
Bin0.bind("<<Binselect>>", go2)
Bin0.grid(row=0, column=1)
E1 = scrolledtext.ScrolledText(top, width=38, height=7)  # 38 20
E1.place(x=209, y=153)
Eb0 = tk.Button(top, text='清空文本框', font=('Arial', 10), width=10, height=1, command=txt_clr, borderwidth=6,
                relief="groove")
Eb0.place(x=385, y=259)
#------------------------------------------------------------------
#数据处理GUI设计
tk.Label(top,text='数据管理:',font=('Arial',10),width=25,
         height=1,
         borderwidth=1,
         relief="groove",bg='green'
         ).grid(row=4,column=0,columnspan=2)
tk.Label(top,text='数据保存路径:',font=('Arial',10),width=12,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=5,column=0,columnspan=1)
eb1=tk.StringVar(value='data.txt')
file_name1=tk.Entry(top,textvariable=eb1,borderwidth=3,relief="groove",bg='cyan',width=12)
file_name1.grid(row=5,column=1,columnspan=4)
b3=tk.Button(top,text='数据删除:',font=('Arial',9),width=10,height=1,command= file_move,borderwidth=5,relief="groove")
b3.grid(row=6,column=0)
eb2=tk.StringVar(value='all')
file_name2=tk.Entry(top,textvariable=eb2,borderwidth=3,relief="groove",bg='cyan',width=12)
file_name2.grid(row=6,column=1,columnspan=4)
b4=tk.Button(top,text='数据处理:',font=('Arial',9),width=10,height=1,command= data_deal,borderwidth=5,relief="groove")
b4.grid(row=7,column=0)
eb3=tk.StringVar(value='None')
file_name3=tk.Entry(top,textvariable=eb3,borderwidth=3,relief="groove",bg='cyan',width=12)
file_name3.grid(row=7,column=1,columnspan=4)
b5=tk.Button(top,text='重置缓冲区',font=('Arial',10),width=10,height=1,command= dat_clr,borderwidth=5,relief="groove")
b5.grid(row=8,column=0,columnspan=10)
#------------------------------------------------------------------
#画图GUI设计
def select3():
    find_flag = 0
    tet = ''
    filename = tk.filedialog.askopenfilename(filetypes=[("text file","split_*.txt")])
    for a in filename :
            if a=='/' :
                    if find_flag==1 :
                            tet=''
                            find_flag=0
            if find_flag==1 :
                    tet=tet+a
            if a=="/" :
                    find_flag = 1
            if a=='.':
                    find_flag=0
    e12=tk.StringVar(value="1")
    if tet!='':
        eb4.set(tet+'txt')
        file_name4['background'] = 'yellow'
    else:
        eb4.set('')
        file_name4['background'] = 'cyan'
tk.Label(top,text='波形分析:',font=('Arial',10),width=25,
         height=1,
         borderwidth=1,
         relief="groove",bg='green'
         ).grid(row=9,column=0,columnspan=2)
bottom3=tk.Button(top,text='画图文件:',font=('Arial',10),width=10,height=1,command=select3,borderwidth=5,relief="groove")
bottom3.grid(row=10,column=0)
b5=tk.Button(top,cursor='star',text='画图',font=('Arial',12),width=10,height=1,command=draw,borderwidth=5,relief="groove",bg='yellow')
b5.grid(row=14,column=1,columnspan=2)
eb4=tk.StringVar(value='split_....txt')
file_name4=tk.Entry(top,textvariable=eb4,borderwidth=3,relief="groove",bg='cyan',width=12)
file_name4.grid(row=10,column=1,columnspan=4)
e6=tk.StringVar(value=0)
tk.Label(top,text='起始条数',font=('Arial',10),width=12,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=11,column=0)
tk.Label(top,text='终止条数',font=('Arial',10),width=12,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=11,column=1)
x_table_start=tk.Entry(top,textvariable=e6,borderwidth=3,relief="groove",bg='cyan',width=12)
x_table_start.grid(row=12,column=0,columnspan=1)
e9=tk.StringVar(value=1000)
x_table_end=tk.Entry(top,textvariable=e9,borderwidth=3,relief="groove",bg='cyan',width=12)
x_table_end.grid(row=12,column=1,columnspan=1)
tk.Label(top,text='数据类型:',font=('Arial',10),width=12,
         height=1,justify=tk.LEFT,
         borderwidth=1,
         relief="groove").grid(row=13,column=0)
def gos(*args):
    print(Bin_type.get())
etype=tk.StringVar(value='int8')
Bin_type=ttk.Combobox(textvariable=etype,width=10)
Bin_type["values"]=('int8','int16','int32','uint8')
Bin_type.current(0)
Bin_type.bind("<<Binselect>>",gos)
Bin_type.grid(row=13,column=1)
def gop(*args):
    print(Bin_pic.get())
epic=tk.StringVar(value='折线图')
Bin_pic=ttk.Combobox(textvariable=epic,width=10)
Bin_pic["values"]=('折线图','散点图')
Bin_pic.current(0)
Bin_pic.bind("<<Binselect>>",gop)
Bin_pic.grid(row=14,column=0)
#示波器GUI设计
tk.Label(top,text='示波器单位值:',font=('Arial',10),width=12,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).place(x=210, y=325)
ebosci=tk.StringVar(value='300')
Bin_range=ttk.Combobox(textvariable=ebosci,width=10)
Bin_range["values"]=('200','900','800','700','600','500','400','300','100')
Bin_range.current(0)
Bin_range.bind("<<Binselect>>")
Bin_range.place(x=350,y=322)
tk.Label(top,text='示波器:',font=('Arial',10),width=34,
         height=1,
         borderwidth=1,
         relief="groove",bg='green'
         ).place(x=210, y=300)
bottom4=tk.Button(top,text='打开示波器',font=('Arial',10),width=10,height=1,command= osci_open,borderwidth=5,relief="groove",bg='yellow')
bottom4.place(x=210, y=385)


tk.Label(top,text='示波器通道:',font=('Arial',10),width=12,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).place(x=210, y=356)
echannel=tk.StringVar(value='L1(电感1)')
Bin_channel=ttk.Combobox(textvariable=echannel,width=10)
Bin_channel["values"]=('L1(电感1)','L2(电感2)','L3(电感3)','L4(电感4)',
                       'L5(电感5)','L6(电感6)','L7(电感7)','pwm(舵机打角)')
Bin_channel.current(0)
Bin_channel.bind("<<Binselect>>")
Bin_channel.place(x=350, y=355)
tk.Label(top,text="e芯: AI Team\n2021.2.16",font=('Arial',10),width=10,height=2,bg='plum')\
    .place(x=352, y=383)
top.mainloop()