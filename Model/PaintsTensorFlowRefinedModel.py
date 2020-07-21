#PaintsTensorflowTraining

from preprocess.Datasets import *
from subnet import *

class PaintsTensorFlowTrain:
    def __init__(self, batch_size, model_name="PaintsTensorFlow"):
        self.data_sets = Datasets_512(batch_size)
	self.batch_size = batch_size
        self.model_name = "{}".format(model_name)
        initdir(self.model_name)

        self.global_steps = tf.compat.v1.train.get_or_create_global_step()
        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.saved_refined = '/saved_model/PaintsTensorFlowRefinedModel.h5' #in case of resume in training
        self.generator_128 =  tf.keras.models.load_model('/saved_model/PaintsTensorFlowDraftModel.h5')  #draft model path
        self.generator_512 = tf.keras.models.load_model(self.saved_refined)   
#         self.generator_512 = Generator(res_net_block=False)    # in case of first time of training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)


    def __loss(self, output, target):
        loss = tf.reduce_mean(tf.abs(target - output))
        return loss

    def __pred_image(self, model, image, line, hint, draft, epoch=None):
        gs = self.global_steps.numpy()
        predImage = model.predict([line, draft])


        if epoch is not None:
            loss = self.__loss(predImage, image)

            loss = "{:0.05f}".format(loss).zfill(7)
            print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.global_steps.numpy(), loss))


        hint = np.array(hint)
        hint[hint > 1] = 1

        lineImage = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([lineImage, hint, draft, predImage, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

        
    def __draft_image(self, line_128, hint_128):
        draft = self.generator_128.predict([line_128, hint_128])
        draft = tf.image.resize(draft, size=(512, 512),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return draft
    
    def fast_fetch(self,size, Traindata):        
        if (size[3]==3):
            x=np.zeros(size)
            for i in range(0,self.batch_size):
                t = Traindata[i]
                x[i] = np.asarray(t, np.float32)        
            return x
        elif (size[3] ==1):
            x=np.zeros(size)
            for i in range(self.batch_size):
                t = Traindata[i]
                x[i,:,:,0] = np.asarray(t,np.float32).reshape(size[1],size[2])
            return x

    def training(self, loadEpochs=0):

        images, lines = self.data_sets.get_image_and_line()

        for epoch in range(loadEpochs):

            print ('epoch {}'.format(epoch))
            print("GS: ", self.global_steps.numpy())
            for batchs in range(0, len(images)/self.batch_size):
                
                print ('Batch no:'+ batchs , end = '\r')
                trainData = self.data_sets._next(images[(batchs*self.batch_size):(batchs*self.batch_size)+self.batch_size], lines[(batchs*self.batch_size):(batchs*self.batch_size)+self.batch_size])
 
                image = tf.convert_to_tensor(self.fast_fetch((self.batch_size,512,512,3), trainData[2]), dtype=tf.float32, dtype_hint=None, name=None)        
                line = tf.convert_to_tensor(self.fast_fetch((self.batch_size,512,512,1), trainData[3]), dtype=tf.float32, dtype_hint=None, name=None)
                hint = tf.convert_to_tensor(self.fast_fetch((self.batch_size,512,512,3), trainData[self.batch_size]), dtype=tf.float32, dtype_hint=None, name=None)
                line_128 = tf.convert_to_tensor(self.fast_fetch((self.batch_size,128,128,1), trainData[0]), dtype=tf.float32, dtype_hint=None, name=None)
                hint_128 = tf.convert_to_tensor(self.fast_fetch((self.batch_size,128,128,3), trainData[1]), dtype=tf.float32, dtype_hint=None, name=None)

                draft = self.__draft_image(line_128, hint_128)
            
                with tf.GradientTape() as tape:
                    genOut = self.generator_512(inputs=[line, draft], training=True)
                    loss = self.__loss(genOut, image)
                gradients = tape.gradient(loss, self.generator_512.variables)
                self.optimizer.apply_gradients(zip(gradients, self.generator_512.variables))

                

                gs = self.global_steps.numpy()
            print ("gen loss {}".format(loss))
            print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(self.epochs.numpy(), gs))
                        
            self.generator_512.summary()
            print(self.global_steps)
    
    def save_refined(self, saved_path):

        self.generator_512.save(saved_path, include_optimizer=False)  # for keras Model


    def make_prediction(self, line128_img, hint128_img, line512_img):

        pred512 = self.__draft_image(line128_img, hint128_img)
        final_pred = self.generator_512.predict([line512_img, pred512])
        return final_pred
    
    def convert_to_pred(self, image_path, line_path):
        
        def fast_fetch(size, data):        
            if (size[2]==3):
                x=np.zeros((self.batch_size,size[0],size[1],size[2]))
                x[0] = np.asarray(data, np.float32)        
                return x
            elif (size[2] ==1):
                x=np.zeros((self.batch_size,size[0],size[1],size[2]))
                x[0,:,:,0] = np.asarray(data,np.float32).reshape(size[0],size[1])
                return x
          
        
        loadData = line_to_data(image_path, line_path)
        line128, hint128, line512 = loadData.convert_to_128()
        line128 = tf.convert_to_tensor(fast_fetch((128,128,1), line128), dtype=tf.float32, dtype_hint=None, name=None)
        hint128 = tf.convert_to_tensor(fast_fetch((128,128,3), hint128), dtype=tf.float32, dtype_hint=None, name=None)
        line512 = tf.convert_to_tensor(fast_fetch((512,512,1), line512), dtype=tf.float32, dtype_hint=None, name=None)
        prediction = self.make_prediction(line128, hint128, line512)
        return prediction
        
        
class line_to_data:
    def __init__(self, image_path, line_path):
        self.line_real_path = line_path
        self.image_path = image_path
        
    def convert2float(self, img):
        img = tf.cast(img, tf.float32)
        img = (img / 127.5)
        return img
        
    def built_hint(self, image):
        random = np.random.rand
        hint = np.ones_like(image)
        hint += 1
        leak_count = np.random.randint(16, 120)
        if random() < 0.4:
            leak_count = 0
        elif random() < 0.7:
            leak_count = np.random.randint(2, 16)
        # leak position
        x = np.random.randint(1, image.shape[0] - 1, leak_count)
        y = np.random.randint(1, image.shape[1] - 1, leak_count)
        def paintCel(i):
            color = image[x[i]][y[i]]
            hint[x[i]][y[i]] = color
            if random() > 0.5:
                hint[x[i]][y[i] + 1] = color
                hint[x[i]][y[i] - 1] = color
            if random() > 0.5:
                hint[x[i] + 1][y[i]] = color
                hint[x[i] - 1][y[i]] = color
        for i in range(leak_count):
            paintCel(i)
        return hint
        
    def convert_to_128(self):
        self.image = tf.io.read_file(self.image_path)
        self.image = tf.image.decode_jpeg(self.image, channels=3)
        self.line_real = tf.io.read_file(self.line_real_path)
        self.line_real = tf.image.decode_jpeg(self.line_real, channels=1)
        self.image_128 = tf.image.resize(self.image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.line_128 = tf.image.resize(self.line_real, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.hint_128 = self.built_hint(self.image_128)
        self.line_512 = tf.image.resize(self.line_real, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.image = self.convert2float(self.image)
        self.line_real = self.convert2float(self.line_real)
        self.line_128 = self.convert2float(self.line_128)
        self.line_512 = self.convert2float(self.line_512)

        return self.line_128, self.hint_128, self.line_512
        

        


        
        
