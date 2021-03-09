import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import ttk
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten,ReLU
from keras.layers import BatchNormalization
import os.path as path
import threading


top=tk.Tk()#创建消息循环
top.title('AI_TRAINER_0.1.1')
top.geometry("920x800+500+200")

def LoadAndSelectData(ad_batch_size):
    #cmd = 'python ./data_convet_seekfree.py -b %d' % ad_batch_size
    #print(cmd)
    #os.popen(cmd + ' > null.log').read()
    train_data = np.load(train_file1.get())
    test_data = np.load(test_file1.get())
    train_label= np.load(train_file2.get())
    test_label= np.load(test_file2.get())
    return train_data,test_data,train_label,test_label
#train_data = np.load('./ad_train_dat.npy')
#    test_data = np.load('./ad_test_dat.npy')

 #   train_label= np.load('./pwm_train_label.npy')
 #   test_label= np.load('./pwm_test_label.npy')

def CreateModelDense(num_classes=1, drop=0.25, isBN=True,ad_batch_size=1):
    isBN = 0
    model = Sequential()
    # -----------------
    model.add(Flatten(input_shape = (ad_batch_size,int(inputdim.get()),1)))

    model.add(Dense(int(H1.get())))
    model.add(Activation('tanh'))
    if isBN:
        model.add(BatchNormalization())
    if drop > 0:
        model.add(Dropout(drop))
    model.add(Dense(int(H2.get())))
    model.add(Activation('tanh'))
    if isBN:
        model.add(BatchNormalization())
    if drop > 0:
        model.add(Dropout(drop))
    #自己加的
    #model.add(Dense(32))
    #model.add(Activation('tanh'))
   # model.add(Dense(75))
   # model.add(Activation('tanh'))
   # model.add(Dense(55))
    #model.add(Activation('tanh'))
    # 自己加的
    model.add(Dense(int(H3.get())))
    model.add(ReLU(max_value=8))
    model.summary()

    #model.add(Dense(int(128)))
    #model.add(Activation('tanh'))
    #model.add(Dense(int(64)))
   # model.add(Activation('tanh'))
    #model.add(Dense(32))
    #model.add(ReLU(max_value=8))

    if isBN:
        model.add(BatchNormalization())
    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    sModelName = 'smartcar_ad_dense_drop_0%d_adSize_%d' % (int(drop * 100),ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model

#listbox=tk.Listbox(top,width=1,height=2)
#listbox.grid(row=0,column=1)
#scale=tk.Scale(top)
#scale.grid(sticky=tk.E,row=0,rowspan=800,column=11,ipady=1000)
#def resize():
#    scale.get()
#scale=tk.Scale(top,from_=10,to=40,orient=tk.HORIZONTAL,command=resize)
#scale.set(0)
#scale.grid(row=0,column=80)


show_mod = 0
def sc_keras(x=1):

    min_loss_cut = 0
    see= tk.Tk()  # 创建消息循环
    see.title('show_mode')
    see.geometry("320x70")
    def func1():
        global show_mod
        show_mod=0
        see.destroy()
    def func2():
        global show_mod
        show_mod=1
        see.destroy()
    def func3():
        global show_mod
        show_mod=2
        see.destroy()

    tk.Label(see, text='选择可视化模式', font=('Arial',10), width=35,
             height=1,borderwidth=8,
                   relief="groove").grid(row=0, column=1,columnspan=70 )
    b4 = tk.Button(see, text='曲线拟合观察',font=('Arial', 10), width=10, height=1, command=func1, borderwidth=5,
                   relief="groove")
    b4.grid(row=2, column=1)

    b5 = tk.Button(see, text='loss曲线', font=('Arial', 10), width=10, height=1, command=func2, borderwidth=5,
                   relief="groove")
    b5.grid(row=2, column=2)
    b6 = tk.Button(see, text='取消', font=('Arial', 10), width=10, height=1, command=func3, borderwidth=5,
                   relief="groove")
    b6.grid(row=2, column=3)
    see.mainloop()
    print('make!')
    filename1=test_file1.get()
    filename2=test_file2.get()
    filename3=train_file1.get()
    filename4=train_file2.get()
    with PdfPages('loss.pdf') as pdf:

        ad_size = int(adsize.get())
        # Load and select data
        x_train, x_test, y_train, y_test = LoadAndSelectData(ad_size)
        screen["text"] += "train files are loaded SUCESSFULLY!!\n"
        lr = float(learn_rate.get())
        minLR =0.0001
        batch_size = 25
        decay = float(1.5 / 1e6)
        histCnt = 20
        lossHist = [1E4] * histCnt
        epocsPerTrain = 1
        burstOftCnt = 0
        minLoss = 1E5
        maxAccu = 0
        opt = keras.optimizers.Adamax(lr, decay)
        if Bin2.get() == 'SGD':
            opt = keras.optimizers.SGD(lr, decay)
        elif Bin2.get() == 'RMSprop':
            opt = keras.optimizers.RMSprop(lr, decay)
        elif Bin2.get() == 'Adamax':
            opt = keras.optimizers.Adamax(lr, decay)
        elif Bin2.get() == 'Adagrad':
            opt = keras.optimizers.Adagrad(lr, decay)
        elif Bin2.get() == 'Adadelta':
            opt = keras.optimizers.Adadelta(lr, decay)
        screen["text"] += "Using "+Bin2.get()+"optimizer\n"
        num_classes = 1
        inputsize=int(inputdim.get())
        outputsize=int(output.get())
        print(x_train)
        print(inputsize)
        print(outputsize)
        x_train = x_train.reshape(int(x_train.size / ad_size / inputsize), ad_size, inputsize, outputsize)
        x_test = x_test.reshape(int(x_test.size / ad_size / inputsize), ad_size, inputsize, outputsize)

        x_train = x_train.astype('int8')
        y_train = y_train.astype('int8')
        x_test = x_train.astype('int8')
        y_test = y_train.astype('int8')
        print('Training data shape:%d' % (min(x_train.flatten())))
        x_train = (x_train / 128).astype('float32')
        x_test = (x_test / 128).astype('float32')
        y_train = ((y_train) / 128).astype('float32')
        y_test = ((y_test) / 128).astype('float32')
        print('x_test data shape:%f~%f' % (max(x_train.flatten()), min(x_train.flatten())))
        print('y_test data shape:%f~%f' % (max(y_train), min(y_train)))


        model_name, model = CreateModelDense(num_classes, float(drop_rate.get()), not 'store_true', ad_size)
        print('Training model ' + model_name)
        screen["text"]+="The Dense Model "+inputdim.get()+"*"+H1.get()+H2.get()+"*"+H3.get()+"*"+output.get()+"is made!\n"
        # train the model using RMSprop
        logFile = model_name + '_log.txt'
        sSaveModel = '%s.h5' % (model_name)
        sSaveCtx = '%s_ctx.h5' % (model_name)

        show=[]
        show.append('模型报告:层数/权重'+'\n')
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()
        for name, weight in zip(names, weights):
            show.append(str(name)+str(weight.shape)+'\n')
            #show.append()
            #show.append('\n')
        show+='请确认模型构架'
        sumar = tk.Tk()  # 创建消息循环
        sumar.title('model_report')
        sumar.geometry("450x580")
        def yy():
            sumar.destroy()
        b7 = tk.Button(sumar, text='确定', font=('Arial', 10), width=10, height=1, command=yy, borderwidth=5,
                       relief="groove").grid(row=2, column=1)
        entry = tk.Label(sumar,text=show,fg='yellow', width=60, height=30,borderwidth=15,relief="groove",bg='green')
        entry.grid(row=1, column=1)


        sumar.mainloop()
        if Bin.get() == 'continue' and path.exists(sSaveCtx):
            fd = open(logFile, 'r')
            s = fd.read()
            fd.close()
            lst = s.split('\n')[-2].split(',')
            for s in lst:
                if s.find('lr=') == 0:
                    lr = float(learn_rate.get())
                    lr *= (1 - decay) ** (50000 / batch_size)############147
                if s.find('times=') == 0:
                    i = int(s[6:]) - 1 + 1
                if s.find('accu=') == 0:
                    maxAccu = float(s[5:])
                if s.find('loss=') == 0:
                    minLoss = float(s[5:])
                    if(se.get()):
                        minLoss=float(lo.get())
            print('resume training from ', lst)
            model = load_model(sSaveCtx)
            fd = open(logFile, 'a')
        else:
            fd = open(logFile, 'w')
            s = 'times=%d,loss=%.4f,accu=%.4f,lr=%f,decay=%f' % (0, minLoss, maxAccu, lr, decay)
            fd.write(s + '\n')
            fd.close()
            fd = open(logFile, 'a')
        i = 0

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        screen["text"] +='learning rate:'+learn_rate.get()+'\n'+'drop_rate:'+drop_rate.get()+'\n'
        lossy = []
        t = []
        over_fit = []
        Acc = []

        
        screen["text"] +="train start!!\n"
        while i < int(epcho.get()):

            top.update()
            print('Train %d times' % (i + 1))
            hist = model.fit(x_train, y_train, batch_size, epochs=epocsPerTrain, \
                             shuffle=True, callbacks=None)  # callbacks=[TensorBoard(log_dir='./log_' + model_name)])
            model.save(sSaveCtx)
            # evaluate
            top.update()
            loss, accuracy = model.evaluate(x_test, y_test)
            # process loss
            lossy.append(hist.history['loss'])
            loss = int(loss * 10000) / 10000.0
            min_loss_cut=min_loss_cut+1


            if loss < minLoss:
                min_loss_cut=0
                minLoss = loss
                maxAccu = accuracy
                print('Saved a better model!')
                model.save(sSaveModel)
                s = 'Saved times=%d,loss=%.4f,accu=%.4f,lr=%f,decay=%f' % (i + 1, minLoss, maxAccu, lr, decay)
            else:
                # save log
                s = 'times=%d,loss=%.4f,accu=%.4f,lr=%f,decay=%f' % (i + 1, minLoss, maxAccu, lr, decay)
            # check if it is overfit
            oftCnt = 0
            for k in range(histCnt):
                if loss > lossHist[k]:
                    oftCnt += 1
            oftRate = oftCnt / histCnt
            print('overfit rate = %d%%' % int(oftRate * 100))
            if oftCnt / histCnt >= 0.6:
                burstOftCnt += 1
                if burstOftCnt > 3:
                    print('Overfit!')
            else:
                burstOftCnt = 0
            s = s + 'overfit rate = %d%%' % int(oftRate * 100)

            lossHistPrnCnt = 6
            if lossHistPrnCnt > histCnt:
                lossHistPrnCnt = histCnt
            fd.write(s + '\n')
            print(s, lossHist[:lossHistPrnCnt])
            fd.close()

            # update loss history
            for k in range(histCnt - 1):
                ndx = histCnt - 1 - k
                lossHist[ndx] = lossHist[ndx - 1]
            lossHist[0] = loss

            fd = open(logFile, 'a')
            lr *= (1 - decay) ** (50000 / batch_size)
            if min_loss_cut==5 :
                if (se2.get()):
                    lr=lr/2
            if lr < minLR:
                lr = minLR
            model = load_model(sSaveCtx)
            opt = keras.optimizers.SGD(lr, decay)
            if Bin2.get()=='SGD':
                opt = keras.optimizers.SGD(lr, decay)
            elif Bin2.get()=='RMSprop':
                opt = keras.optimizers.RMSprop(lr, decay)
            elif Bin2.get() == 'Adamax':
                opt = keras.optimizers.Adamax(lr, decay)
            elif Bin2.get() == 'Adagrad':
                opt = keras.optimizers.Adagrad(lr, decay)
            elif Bin2.get() == 'Adadelta':
                opt = keras.optimizers.Adadelta(lr, decay)
            print('new lr_rate=%f' % (lr))
            #'SGD','RMSprop','Adam','Adagrad','Adadelta'
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
            i += 1
            # if i%15==0 :
            # np_loss=np.array(lossy)
            top.update()
            t.append(i - 1)
            over_fit.append(oftRate * 100)
            Acc.append(accuracy)
            if i%26==0:
                screen["text"]=' '
            screen["text"] +=str(i)+':   '+str(minLoss)+'  '+'lr:'+str(lr)+'\n'
            if show_mod==1:

                plt.ion()
                plt.figure(1, facecolor='#FFDAB9')
                plt.ylabel('loss', fontsize=14)
                plt.subplot(212, facecolor='cyan')
                plt.grid()
                plt.tick_params(labelsize=7, color='#FFA500')
                    #  plt.xlim(-1,120)
                plt.xlabel('times%*.d' % (10, i), fontsize=10)
                plt.title('loss%*.4f' % (10, loss), fontsize=12, color='midnightblue')
                plt.plot(t, lossy, '-b', marker='s', markersize=1.7)

                plt.subplot(222, facecolor='cyan')
                plt.grid()
                plt.tick_params(labelsize=6)
                    # plt.xlabel('times%*.d' % (10, i - 1), fontsize=10)
                plt.title('accuracy%*.4f' % (10, accuracy), fontsize=10, color='midnightblue')
                plt.plot(t, Acc, '-g')

                plt.subplot(221, facecolor='cyan')
                plt.grid()
                plt.tick_params(labelsize=6)
                    # plt.xlabel('times%*.d' % (10, i - 1), fontsize=10)
                co = 'midnightblue'
                if (oftRate * 100 > 50):
                    co = '#FF0000'
                plt.title('overfit rate = %d%%' % int(oftRate * 100), fontsize=10, color=co)
                plt.plot(t, over_fit, '-r')
                if i % 15 == 0:
                    pdf.savefig()
                plt.draw()
                plt.pause(0.0001)
            if show_mod == 0:

                test_label = np.load(test_file2.get())
                test_label = test_label.astype('int8')
                test_label = (test_label / 128).astype('float32')

                test_data = np.load(test_file1.get())
                test_data = test_data.reshape(int(test_data.size / 1 / int(inputdim.get())), 1, int(inputdim.get()), 1)

                test_data = test_data.astype('int8')
                test_data = (test_data / 128).astype('float32')

                data = test_data[900:1000]

                forward = Model(inputs=model.input, outputs=model.output)
                out = forward.predict(data)
                out2 = out.flatten()

                y_test_lable = test_label[900:1000]
                # test_dat_orgin= test_dat_orgin[int(start.get()):int(number.get())]
                x = 100
                y_lab = out2[:x]
                X = np.arange(0, x, 1)
                plt.ion()
                plt.figure(2,figsize=(10,5))
                plt.cla()
                plt.subplot(111, facecolor='cyan')
                plt.title('Compare——loss:'+str(loss))
                plt.tick_params(labelsize=6)
                plt.plot(X, y_test_lable)
                plt.plot(X, y_lab)
                plt.draw()
                plt.pause(0.0001)






t1 = threading.Thread(target=sc_keras, args=("t1",))


def go():
   t1.start()



tk.Label(top,text='// AI  训  练  手 //',font=('Arial',12),width=60,
         height=2,
         borderwidth=3,
         bg='green',
         relief="groove").grid(row=1,column=0,columnspan=3)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=2,column=0,columnspan=3)


tk.Label(top,text='参 数 设 置 ',font=('Helvetica',10),width=60,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove",
         ).grid(row=3,column=0,columnspan=3)
tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=4,column=0,columnspan=3)
tk.Label(top,text='学习率 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=5,column=0)

e0=tk.StringVar(value=0.01)
learn_rate=tk.Entry(top,width=14,textvariable=e0,borderwidth=2,relief="groove",bg='cyan')
learn_rate.grid(row=5,column=1)


tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=6,column=0,columnspan=3)
tk.Label(top,text='丢弃率 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=7,column=0)
e1=tk.StringVar(value=0.25)
drop_rate=tk.Entry(top,width=14,textvariable=e1,borderwidth=2,relief="groove",bg='cyan')
drop_rate.grid(row=7,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=8,column=0,columnspan=3)
tk.Label(top,text='批尺寸 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=9,column=0)
e2=tk.StringVar(value=1)
adsize=tk.Entry(top,width=14,textvariable=e2,borderwidth=2,relief="groove",bg='cyan')
adsize.grid(row=9,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=10,column=0,columnspan=3)
tk.Label(top,text='输入维度 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=11,column=0)
e3=tk.StringVar(value=7)
inputdim=tk.Entry(top,width=14,textvariable=e3,borderwidth=2,relief="groove",bg='cyan')
inputdim.grid(row=11,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=12,column=0,columnspan=3)
tk.Label(top,text='第一隐层 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=13,column=0)
e4=tk.StringVar(value=140)
H1=tk.Entry(top,width=14,textvariable=e4,borderwidth=2,relief="groove",bg='cyan')
H1.grid(row=13,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=14,column=0,columnspan=3)
tk.Label(top,text='第二隐层 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=15,column=0)
e5=tk.StringVar(value=100)
H2=tk.Entry(top,width=14,textvariable=e5,borderwidth=2,relief="groove",bg='cyan')
H2.grid(row=15,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=16,column=0,columnspan=3)
tk.Label(top,text='第三隐层 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=17,column=0)
e6=tk.StringVar(value=40)
H3=tk.Entry(top,width=14,textvariable=e6,borderwidth=2,relief="groove",bg='cyan')
H3.grid(row=17,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=18,column=0,columnspan=3)
tk.Label(top,text='输出维度 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=19,column=0)
e7=tk.StringVar(value=1)
output=tk.Entry(top,width=14,textvariable=e7,borderwidth=2,relief="groove",bg='cyan')
output.grid(row=19,column=1)

tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=20,column=0,columnspan=3)


tk.Label(top,text='数  据  配  置 ',font=('Arial',10),width=60,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove",
         ).grid(row=21,column=0,columnspan=3)
tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=22,column=0,columnspan=3)
tk.Label(top,text='训练集 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=23,column=0)
e8=tk.StringVar(value='./ad_train_dat.npy')
train_file1=tk.Entry(top,textvariable=e8,borderwidth=2,relief="groove",bg='cyan')
train_file1.grid(row=23,column=1)
e9=tk.StringVar(value='./pwm_train_label.npy')
train_file2=tk.Entry(top,textvariable=e9,borderwidth=2,relief="groove",bg='cyan')
train_file2.grid(row=24,column=1)
tk.Label(top,text='------------------------------------------------------'
                  '-----------------------------------------------------',font=('Arial',12),width=60,
         height=1,).grid(row=25,column=0,columnspan=3)
tk.Label(top,text='测试集 :',font=('Arial',10),width=10,justify=tk.LEFT,
         height=1,
         borderwidth=1,
         relief="groove"
         ).grid(row=26,column=0)
e10=tk.StringVar(value='./ad_test_dat.npy')
test_file1=tk.Entry(top,textvariable=e10,borderwidth=2,relief="groove",bg='cyan')
test_file1.grid(row=26,column=1)
e11=tk.StringVar(value='./pwm_test_label.npy')
test_file2=tk.Entry(top,textvariable=e11,borderwidth=2,relief="groove",bg='cyan')
test_file2.grid(row=27,column=1)

b3=tk.Button(top,text='开始训练/epchos',fg='yellow',bg='green',font=('Arial',12),width=15,height=1,command= go,borderwidth=5,relief="groove")
b3.grid(row=29,column=0,columnspan=1)
photo=tk.PhotoImage(file=r'1.png')
tk.Label(top,image=photo,justify=tk.LEFT,borderwidth=4,
         relief="solid").grid(row=1,column=4,rowspan=4)
screen=tk.Label(top,text='Welcome to AI Team!!!\n',font=('Arial',10),width=38,height=32,justify=tk.LEFT,fg='blue',bg='lightgreen',borderwidth=5,relief="groove")
screen.grid(row=2,column=4,columnspan=5,rowspan=50)

#screen=tk.Label(top,text="Welcome to AI Team !!!",font=('Arial',10),width=30,justify=tk.LEFT,fg='blue',bg='lightgreen',
#         height=50,
 #        borderwidth=3,
 #        relief="groove"
   #      ).grid(row=0,column=4,columnspan=5,rowspan=50)

tk.Label(top,text='Team: Yukun XueJinBo GeWenJie',font=('Arial',8),width=40,bg='plum',
            borderwidth=5,
         relief="groove",
         height=1,).grid(row=31,column=0)




def go1(*args):
    print(Bin.get())
e12=tk.StringVar(value='continue')
Bin=ttk.Combobox(textvariable=e12,width=12)
Bin["values"]=('continue','restart')
Bin.current(0)
Bin.bind("<<Binselect>>",go1)
Bin.grid(row=29,column=1)

e14=tk.StringVar(value=120)
epcho=tk.Entry(top,textvariable=e14,borderwidth=2,relief="groove",bg='cyan')
epcho.grid(row=30,column=0)

def go2(*args):
    print(Bin2.get())
e13=tk.StringVar(value='SGD')
Bin2=ttk.Combobox(textvariable=e13,width=12)
Bin2["values"]=('Adamax','SGD','RMSprop','Adagrad','Adadelta')
Bin2.current(0)
Bin2.bind("<<Binselect>>",go2)
Bin2.grid(row=30,column=1)
tk.Label(top,text='e芯实验室 2020年4月11日',font=('Arial',8),width=70,
         height=1,).grid(row=32,column=0)




se = tk.IntVar(0)
w1=tk.Checkbutton(top, text='重新设置最小loss'
                                , variable=se).grid(row=30,column=4)

e16=tk.StringVar(value='0.100')
lo=tk.Entry(top,textvariable=e16,borderwidth=1,relief="groove",bg='white')
lo.grid(row=31,column=4)

se2 = tk.IntVar(0)
w2=tk.Checkbutton(top, text='是否二分法调整学习率'
                                , variable=se2,fg='red').grid(row=29,column=4)
top.mainloop()