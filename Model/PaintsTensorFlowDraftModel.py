#PaintsTensorflowDraftModel
from preprocess.Datasets import *
from subnet import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)

class PaintsTensorFlowDraftModelTrain:
    def __init__(self,batch_size, model_name="PaintsTensorFlowDraftModel"):
	self.batch_size = batch_size
        self.data_sets = Datasets(self.batch_size)
        self.model_name = model_name
        # utils.initdir(self.model_name)

        self.global_steps = tf.compat.v1.train.get_or_create_global_step()
        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

        self.generator = Generator(name="PaintsTensorFlowDraftNet")  # for first time of training
        self.generator = tf.keras.models.load_model(__SAVED_MODEL_PATH__)  # to resume training 
        
        self.discriminator = Discriminator()

    def __discriminator_loss(self, real, fake):
        SCE = tf.nn.sigmoid_cross_entropy_with_logits
        self.real_loss = SCE(tf.ones_like(real), logits=real)
        self.fake_loss = SCE(tf.zeros_like(fake), logits=fake)
        loss = self.real_loss + self.fake_loss
        return loss

    def __generator_loss(self, disOutput, output, target):
        SCE = tf.nn.sigmoid_cross_entropy_with_logits
        self.gan_loss = SCE(tf.ones_like(disOutput), logits=disOutput)
        self.image_loss = tf.reduce_mean(tf.abs(target - output)) * l1_scaling
        loss = self.image_loss + self.gan_loss
        return loss

    def __pred_image(self, model, image, line, hint, epoch=None):
        global_steps = self.global_steps.numpy()
        pred_image = model.predict([line, hint])

        zero_hint = tf.ones_like(hint)
        zero_hint += 1
        pred_image_zero = model.predict([line, zero_hint])

        dis_fake = self.discriminator(pred_image, training=False)
        loss = self.__generator_loss(dis_fake, pred_image, image)


        loss = "{:0.05f}".format(loss).zfill(7)
        print("Epoch:{} GS:{} LOSS:{}".format(epoch, global_steps, loss))

        hint = np.array(hint)
        hint[hint > 1] = 1

        line_image = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([line_image, hint, pred_image_zero, pred_image, image], 1)
        save_img = utils.convert2uint8(save_img)
        
    def training(self, image_path, line_path, loadEpochs=0):

        images, lines = self.data_sets.get_image_and_line(image_path, line_path)
        batch_steps = len(images) / self.batch_size
        def fast_fetch(size, Traindata):        
            if (size==(self.batch_size,128,128,3)):
                x=np.zeros(size)
                for i in range(0,self.batch_size):
                    t = Traindata[i]
                    x[i] = np.asarray(t, np.float32)        
                return x
            else:
                x=np.zeros(size)
                for i in range(self.batch_size):
                    t = Traindata[i]
                    x[i,:,:,0] = np.asarray(t,np.float32).reshape(128,128)
                return x
            
        for epoch in range(loadEpochs):

            print("GS: ", self.global_steps.numpy(), "Epochs:  ", self.epochs.numpy())
            for batch in range(batch_steps):
                print ('Batch no:'+ batch, end='\r')
                trainData = self.data_sets._next(images[(batch*self.batch_size):(batch*self.batch_size)+self.batch_size], lines[(batch*self.batch_size):(batch*self.batch_size)+self.batch_size])
                image = tf.convert_to_tensor(fast_fetch((self.batch_size,128,128,3), trainData[0]), dtype=tf.float32, dtype_hint=None, name=None)        
                line = tf.convert_to_tensor(fast_fetch((self.batch_size,128,128,1), trainData[1]), dtype=tf.float32, dtype_hint=None, name=None)
                hint = tf.convert_to_tensor(fast_fetch((self.batch_size,128,128,3), trainData[2]), dtype=tf.float32, dtype_hint=None, name=None)
                del (trainData)
                
                with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
                    pred_image = self.generator(inputs=[line, hint], training=True)
                    dis_real = self.discriminator(inputs=image, training=True)
                    dis_fake = self.discriminator(inputs=pred_image, training=True)
                    generator_loss = self.__generator_loss(dis_fake, pred_image, image)
                    discriminator_loss = self.__discriminator_loss(dis_real, dis_fake)
                discriminator_gradients = discTape.gradient(discriminator_loss, self.discriminator.variables)
                generator_gradients = genTape.gradient(generator_loss, self.generator.variables)
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.variables))
                gs = self.global_steps.numpy()
                del(image)
                del(line)
                del(hint)

            self.epochs = self.epochs + 1
            print ('LOSS_G {}\nLOSS_G_Image {}\nLOSS_G_GAN {} \nLOSS_D {}\nLOSS_D_Real {}\nLOSS_D_Fake {}'.format(generator_loss,self.image_loss,self.gan_loss,discriminator_loss,self.real_loss,self.fake_loss))
        self.generator.summary()
    def save_model(self, saved_path):
        self.generator.save(saved_path, include_optimizer=False)  # for keras Model
