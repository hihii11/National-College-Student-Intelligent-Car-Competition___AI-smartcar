#!/usr/bin/python
#-*- coding:UTF-8 -*-
import tkinter as tk
import os
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import ttk
import datetime
from shutil import copyfile
import tkinter.filedialog
top=tk.Tk()#创建消息循环
top.title('DATA_MAKER_0.1.1')
top.geometry("520x420")


a1=tk.StringVar(value='Welcome to AI Team!!!')

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


# extend array, continual 10 ad value
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



str2=' '
filename=' '
def split():
        filenum = 0
        fil1 = file_name1.get()
        fil2 = file_name2.get()
        fil3 = file_name3.get()
        if fil1!='':
                filenum += 1
        if fil2!='':
                filenum += 1
        if fil3 != '':
                filenum += 1

        str1 = ""
        str3 = ""
        str4 = ""
        num = 0
        num1=tezhen.get()
        num2=biaoqian.get()
        textnum = int(tezhen.get())-int('0')+int(biaoqian.get())-int('0')+3
        for k in range(0,filenum):
                str1 = ""
                str3 = ""
                str4 = ""
                if k==0 :
                        filename=fil1
                        fp1 = open(filename, 'r+')
                        str2 = fp1.read()
                if k==1 :
                        filename=fil2
                        fp1 = open(filename, 'r+')
                        str2 = fp1.read()
                if k==2 :
                        filename=fil3
                        fp1 = open(filename, 'r+')
                        str2 = fp1.read()

                for i in str2:
                        if i == ' ':  # 找到空格
                                num += 1  # 数量+1
                                str3 += str1 + i

                                if str1 == "5A":  # 找到固定数据
                                        if num < textnum:  # 数据不够 继续统计
                                                pass
                                        elif num == textnum:  # 找到一组数据
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
                screen["text"] ='Welcome to AI Team!!!\n The file split SUCCESSFULLY!!ლ(╹◡╹ლ)'

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

        ad_dat = ad_array
        pwm_dat = pwm_array

        out_len = int(len(ad_dat) / 100) * 100
        test_ad_npy = ad_dat[0:out_len]
        label_pwm_npy = pwm_dat[0:out_len]
        rest_train = ad_dat[out_len:len(ad_dat)]
        rest_label = pwm_dat[out_len:len(ad_dat)]

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
        screen["text"]=('Welcome to AI Team!!!\n The file concatenate SUCCESSFULLY!!ლ(╹◡╹ლ)')

def convert():
        filenum=0
        fil1 = file_name1.get()
        fil2 = file_name2.get()
        fil3 = file_name3.get()
        if fil1 != '':
                filenum += 1
        if fil2 != '':
                filenum += 1
        if fil3 != '':
                filenum += 1
        batch_size = 1
        # args.file需要打开的文件 dtype设置数据类型 delimiter分隔符 unpack是否每一列当做一个向量输出
        for i in range(0,filenum):
                if i==0:
                        file_name='split_'+file_name1.get()

                if i==1:
                        file_name ='split_'+file_name2.get()
                if i==2:
                        file_name ='split_'+file_name3.get()
                dat_str = np.loadtxt(file_name, dtype=np.str, delimiter=' ', unpack=False)

                # 转换为一维数组 [1 2 3 4 5]
                dat_str = dat_str.reshape(dat_str.size)

                # 定义一个list
                dat_int = []
                # 遍历字符串读取每个数据

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

                # 遍历dat_int矩阵
                for a in dat_int:
                        if (a[10] != 90):
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
                screen["text"]='Welcome to AI Team!!!\n The file converttest SUCCESSFULLY!!ლ(╹◡╹ლ)'
def draw():
        data = np.loadtxt(file_name4.get(), dtype=np.str, delimiter=' ', unpack=False)
        x=int(x_table_end.get())-int((x_table_start.get()))

        data = data.reshape(data.size)
        dat_int = []

        for a in data:

                if (len(a) > 0):
                        hex_str = '0x' + a  # 将数据转换为 0xXX
                        dat_int.append(int(hex_str, 16))

        data = np.array(dat_int)
        # 将数据变形为矩阵 也可以说是二维数组
        data = data.reshape(int(data.size / 11), 11)
        data = data.astype(Bin.get())
#        data = data.astype(ac)
        with PdfPages('induce_pdf.pdf') as pdf:
                x1 = data[int(x_table_start.get()):int(x_table_end.get()), 2]
                x2 = data[int(x_table_start.get()):int(x_table_end.get()),  3]
                x3 = data[int(x_table_start.get()):int(x_table_end.get()),  4]
                x4 = data[int(x_table_start.get()):int(x_table_end.get()),  5]
                x5 = data[int(x_table_start.get()):int(x_table_end.get()),  6]
                x6 = data[int(x_table_start.get()):int(x_table_end.get()),  7]
                x7 = data[int(x_table_start.get()):int(x_table_end.get()),  8]
                x8 = data[int(x_table_start.get()):int(x_table_end.get()),  9]

                x = np.arange(0.0, x, 1)
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce1')
                plt.plot(x, x1)
                pdf.savefig()
                # plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce2')
                plt.plot(x, x2)
                pdf.savefig()
                # plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce3')
                plt.plot(x, x3)
                pdf.savefig()
                # plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce4')
                plt.plot(x, x4)
                pdf.savefig()
                # plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce5')
                plt.plot(x, x5)
                pdf.savefig()
                # plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce6')
                plt.plot(x, x6)
                pdf.savefig()
                # plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.title('induce7')
                plt.plot(x, x7)
                pdf.savefig()
                # plt.show()
                plt.close()

                plt.figure(1)
                plt.subplot(211)
                plt.title('PWM_out')
                plt.plot(x, x8,'.',markersize=1)
                pdf.savefig()
                #plt.close()

                #plt.show()
                plt.close()
                plt.figure(1)
                plt.subplot(212)
                plt.plot(x, x1)
                plt.text(0,x1[1], 'L1', ha='center', va='bottom', fontsize=12)
                plt.plot(x, x2)
                plt.text(0, x2[1], 'L2', ha='center', va='bottom', fontsize=12)
                plt.plot(x, x3)
                plt.text(0, x3[1], 'L3', ha='center', va='bottom', fontsize=12)
                plt.plot(x, x4)
                plt.text(0, x4[1], 'L4', ha='center', va='bottom', fontsize=12)
                plt.plot(x, x5)
                plt.text(0, x5[1], 'L5', ha='center', va='bottom', fontsize=12)
                plt.plot(x, x6)
                plt.text(0, x6[1], 'L6', ha='center', va='bottom', fontsize=12)
                plt.plot(x, x7)
                plt.text(0, x7[1], 'L7', ha='center', va='bottom', fontsize=12)
                plt.title('induce')
                pdf.savefig()
                plt.subplot(211)
                plt.title('PWM_out')
                plt.plot(x, x8, '.', markersize=1)
                premax = 0
                premin = 1000
                for a in x8:
                        if a > premax:
                                premax = a
                        if a < premin:
                                premin = a
                for a,b in zip(x,x8):
                        if a%50==0:
                                plt.text(a,b,(a,b),ha='center',va='bottom',fontsize=8)
                maxderror=0
                prePWM=x8[0]
                for a in  x8:
                        if abs(a-prePWM)>maxderror :
                                maxderror=abs(a-prePWM)
                screen[
                        "text"] = "Welcome to AI Team!!!\n The data draw SUCCESSFULLY!!ლ(╹◡╹ლ)" + "\n" + "PWM最大/最小:" + " " + str(
                        premax) + " " + str(premin)+" 最大斜率"+" "+str(maxderror)
                plt.show()



def set():


        win = tk.Tk()  # 创建消息循环
        win.title('higher_set')
        win.geometry("180x220")


        def sel1():
                var1.set(not var1.get())

        tk.Label(win, text="数据制作高级设置",borderwidth=2,relief="groove").grid(row=0,columnspan=3,)
        var1 = tk.IntVar()
        w1=tk.Checkbutton(win, text='是否重新制作数据        '
                                , variable=var1,command=sel1).grid(row=1)

        def sel2():
                var2.set(not var2.get())
                if var2.get()==1:
                        tk.Label(win,text="将创建数据库AI_data_base").grid(row=4)
                else:
                        tk.Label(win, text="                                        ").grid(row=4)

        tk.Label(win, text="数据库高级设置",borderwidth=2,relief="groove").grid(row=2,columnspan=3)
        var2 = tk.IntVar()
        w2=tk.Checkbutton(win, text='是否建立新数据库         ', variable=var2,command=sel2).grid(row=3,columnspan=3)

        def sel3():
                var3.set(not var3.get())

        var3 = tk.IntVar()
        w3 = tk.Checkbutton(win, text='将此次数据更新到数据库', variable=var3, command=sel3).grid(row=5)

        def sel4():
                var4.set(not var4.get())


        var4 = tk.IntVar()
        w4 = tk.Checkbutton(win, text='更新后打开数据库         ', variable=var4, command=sel4).grid(row=6)
        def applyca():
                if var1.get()==1:
                        path = "."

                        files = os.listdir(path)
                        for file in files:
                                if (file.split(".")[-1] == 'npy'):
                                        if (file[0:9] == 'origin_ad'):
                                               os.remove(file)
                                        elif (file[0:10] == 'origin_pwm'):
                                               os.remove(file)
                                        elif (file[0:2]=='ad'):
                                                os.remove(file)
                                        elif (file[0:3]=='pwm'):
                                                os.remove(file)
                                if (file.split(".")[-1] == 'txt'):
                                        if(file[0:8]!='smartcar'):
                                                os.remove(file)
                if var2.get() == 1:
                        a=os.path.exists('AI_data_base')
                        print(a)
                        if a==1 :
                                print('数据库已经建立')
                                screen["text"] = "Welcome to AI Team!!!\n 数据库已经建立"
                        if a==0 :
                                os.mkdir('AI_data_base')
                                screen["text"] = "Welcome to AI Team!!!\n 数据库建立成功"

                if var3.get()==1:
                        path1=r'./AI_data_base'

                        today=datetime.datetime.today()
                        name1=str(today.year)+'-'+str(today.month)+'-'+str(today.day)
                        path2 = ('AI_data_base\\'+name1)
                        a=os.path.exists(path2)
                        if a==0:
                                print(a)
                                name='./'+str(today.year)+'-'+str(today.month)+'-'+str(today.day)
                                os.mkdir(path1+name)
                        if a==1:
                                screen["text"] = "Welcome to AI Team!!!\n 文件夹已存在"
                        fro_path=file_name1.get()
                        to_path='.\AI_data_base\\'+name1+"\\"+fro_path
                        copyfile(fro_path,to_path)

                        fro_path = file_name2.get()
                        to_path = '.\AI_data_base\\' + name1 + "\\" + fro_path
                        copyfile(fro_path, to_path)

                        fro_path = file_name3.get()
                        to_path = '.\AI_data_base\\' + name1 + "\\" + fro_path
                        copyfile(fro_path, to_path)

                        screen["text"] = "Welcome to AI Team!!!\n 数据库更新"

                if var4.get()==1:
                        today = datetime.datetime.today()
                        name1 = str(today.year) + '-' + str(today.month) + '-' + str(today.day)
                        os.system('start explorer %s'% '.\AI_data_base\\'+name1)
                win.destroy()







        b00 = tk.Button(win, cursor='star', text='Apply', font=('Arial', 12), width=10, height=1, command=applyca,
                       borderwidth=5, relief="groove")
        b00.grid(row=8, column=0, columnspan=3, sticky=tk.W, padx=3)
        win.mainloop()


eb0=tk.StringVar(value='normal.txt')
def select1():
    find_flag = 0
    tet = ''
    filename = tk.filedialog.askopenfilename(filetypes=[("text file","*.txt")])
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
    if tet != '':
            eb0.set(tet + 'txt')
            file_name1['background']='yellow'
    else:
            eb0.set('')
            file_name1['background'] ='cyan'
file_name1=tk.Entry(top,textvariable=eb0,borderwidth=2,relief="groove",bg='cyan')
file_name1.grid(row=1,column=1,padx=3,columnspan=1)
bottom1=tk.Button(top,text='文件1:',font=('Arial',10),width=10,height=1,command=select1,borderwidth=5,relief="groove")
bottom1.grid(row=1,column=0,columnspan=1)
#e0=tk.StringVar(value='normal.txt')
#tk.Label(top,text='文件1:',font=('Arial',10),width=6,height=2).grid(row=1,column=0,columnspan=1)
#file_name1=tk.Entry(top,textvariable=e0,borderwidth=2,relief="groove",bg='cyan')
#file_name1.grid(row=1,column=1,padx=3,columnspan=1)

eb1=tk.StringVar(value='')
def select2():
    find_flag = 0
    tet = ''
    filename = tk.filedialog.askopenfilename(filetypes=[("text file","*.txt")])
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
    if tet != '':
            eb1.set(tet + 'txt')
            file_name2['background'] = 'yellow'
    else:
            eb1.set('')
            file_name2['background'] = 'cyan'
file_name2=tk.Entry(top,textvariable=eb1,borderwidth=2,relief="groove",bg='cyan')
file_name2.grid(row=2,column=1,padx=3,columnspan=1)
bottom2=tk.Button(top,text='文件2:',font=('Arial',10),width=10,height=1,command=select2,borderwidth=5,relief="groove")
bottom2.grid(row=2,column=0,columnspan=1)


#e1=tk.StringVar(value='random.txt')
#tk.Label(top,text='文件2:',font=('Arial',10),width=6,height=2).grid(row=2,column=0,columnspan=1)
#file_name2=tk.Entry(top,textvariable=e1,borderwidth=2,relief="groove",bg='cyan')
#file_name2.grid(row=2,column=1,padx=3,columnspan=1)

eb2=tk.StringVar(value='')
def select3():
    find_flag = 0
    tet = ''
    filename = tk.filedialog.askopenfilename(filetypes=[("text file","*.txt")])
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
        eb2.set(tet+'txt')
        file_name3['background'] = 'yellow'
    else:
        eb2.set('')
        file_name3['background'] = 'cyan'
file_name3=tk.Entry(top,textvariable=eb2,borderwidth=2,relief="groove",bg='cyan')
file_name3.grid(row=3,column=1,padx=3,columnspan=1)
bottom3=tk.Button(top,text='文件3:',font=('Arial',10),width=10,height=1,command=select3,borderwidth=5,relief="groove")
bottom3.grid(row=3,column=0,columnspan=1)


#e3=tk.StringVar(value='manual.txt')
#tk.Label(top,text='文件3:',font=('Arial',10),width=6,height=2).grid(row=3,column=0,columnspan=1)
#file_name3=tk.Entry(top,textvariable=e3,borderwidth=2,relief="groove",bg='cyan')
#file_name3.grid(row=3,column=1,padx=3,columnspan=1)

tk.Label(top,text='特征数:',font=('Arial',10),width=10,height=2).grid(row=1,column=1,columnspan=5)
e4=tk.StringVar(value=7)
tezhen=tk.Entry(top,textvariable=e4,borderwidth=2,relief="groove",bg='cyan')
tezhen.grid(row=1,column=4,columnspan=1)

tk.Label(top,text='标签数:',font=('Arial',10),width=10,height=2).grid(row=2,column=1,columnspan=5)
e5=tk.StringVar(value=1)
biaoqian=tk.Entry(top,textvariable=e5,borderwidth=2,relief="groove",bg='cyan')
biaoqian.grid(row=2,column=4,columnspan=1)

b0=tk.Button(top,cursor='star',text='数据分行!',font=('Arial',12),width=10,height=1,command=split,borderwidth=5,relief="groove")
b0.grid(row=4,column=1,columnspan=3,sticky=tk.W,padx=3)

b1=tk.Button(top,cursor='star',text='数据切片!',font=('Arial',12),width=10,height=1,command=convert,borderwidth=5,relief="groove")
b1.grid(row=4,column=2,sticky=tk.W,padx=3)

b2=tk.Button(top,cursor='star',text='合成文件!',font=('Arial',12),width=10,height=1,command=con,borderwidth=5,relief="groove")
b2.grid(row=4,column=4,columnspan=20,sticky=tk.W,padx=3)

#tk.Label(top,text='画图文件:',font=('Arial',10),width=10,height=2).grid(row=5,column=0)
#e6=tk.StringVar(value='split_normal.txt')
#file_name4=tk.Entry(top,textvariable=e6,borderwidth=2,relief="groove",bg='cyan')
#file_name4.grid(row=5,column=1)
eb22=tk.StringVar(value='split_normal.txt')
def select4():
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
        eb22.set(tet+'txt')
        file_name4['background'] = 'yellow'
    else:
        eb22.set('')
        file_name4['background'] = 'cyan'
file_name4=tk.Entry(top,textvariable=eb22,borderwidth=2,relief="groove",bg='cyan')
file_name4.grid(row=5,column=1)
bottom3=tk.Button(top,text='画图文件:',font=('Arial',10),width=10,height=1,command=select4,borderwidth=5,relief="groove")
bottom3.grid(row=5,column=0)


tk.Label(top,text='显示数:',font=('Arial',10),width=10,height=2).grid(row=5,column=2)
e6=tk.StringVar(value=0)
x_table_start=tk.Entry(top,textvariable=e6,borderwidth=2,relief="groove",bg='cyan')
x_table_start.grid(row=5,column=4)
e9=tk.StringVar(value=1000)
x_table_end=tk.Entry(top,textvariable=e9,borderwidth=2,relief="groove",bg='cyan')
x_table_end.grid(row=6,column=4)

b3=tk.Button(top,cursor='star',text='画图',font=('Arial',12),width=10,height=1,command=draw,borderwidth=5,relief="groove")
b3.grid(row=7,column=3,columnspan=2)

def go(*args):
        print(Bin.get())

e7=tk.StringVar(value='int8')
Bin=ttk.Combobox(textvariable=e7)
Bin["values"]=('int8','int16','int32','uint8')
Bin.current(0)
Bin.bind("<<Binselect>>",go)
Bin.grid(row=7,column=0,columnspan=2)

b4=tk.Button(top,cursor='star',text='高级设置',font=('Arial',12),width=10,height=1,command=set,borderwidth=5,relief="groove")
b4.grid(row=8,column=0,columnspan=2)


screen=tk.Label(top,text='Welcome to AI Team!!!',font=('Arial',10),width=63,height=5,fg='blue',bg='green',borderwidth=5,relief="groove")
screen.grid(row=11,column=0,rowspan=2,columnspan=20,sticky=tk.E)
tk.Label(top,text="AI Team:YuKun XueJinbo GeWenJie ZhengJunTai LiQian",font=('Arial',10),width=60,height=2).grid(row=14,column=0,rowspan=3,columnspan=20)
top.mainloop()

