#Datasets.py
class Datasets:
    def __init__(self, prefetch=-1, batch_size=1, shuffle=False):
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def get_image_and_line(self, image_path, line_path):
        image = glob(image_path)
        image.sort()
        line = glob(line_path)
        line.sort()
        return image, line
        
    def _next(self, image, line):   
        return self.buildDataSets(image, line,'draft')
        

    def _preprocess(self, image, line, training = 'True'):
        if training == 'True':
            if np.random.rand() < 0.5:
                image = cv2.flip(np.float32(image), 0)
                line = cv2.flip(np.float32(line), 0)

            if np.random.rand() < 0.5:
                image = cv2.flip(np.float32(image), 1)
                line = cv2.flip(np.float32(line), 1)

        return image, line, self._buildHint_resize(image)

    def _buildHint_resize(self, image):
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

    def convert2float(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) 
        return image

    def __line_threshold(self, line):
        if np.random.rand() < 0.3:
            line = np.reshape(line, newshape=(512, 512))
            _, line = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            line = np.reshape(line, newshape=(512, 512, 1))
        return line

    def loadImage(self, imagePath, linePath, isTrain='True'):

        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)

        line = tf.io.read_file(linePath)
        line = tf.image.decode_jpeg(line, channels=1)

        image = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        line = tf.image.resize(line, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image = self.convert2float(image)
        line = self.convert2float(line)

        image, line, hint = tf.py_function(self._preprocess,
                                       [np.float32(image), np.float32(line), str(isTrain)],
                                       [tf.float32, tf.float32, tf.float32])
    

        return image, line, hint

    def buildDataSets(self, image, line, mode):
        def build_dataSets(image, line, mode, shuffle=False, isTrain=False):

            if shuffle is False and isTrain is False:
                image.reverse()
                line.reverse()
  
            datasets = tf.data.Dataset.from_tensor_slices((image, line))

	    if mode == 'draft':
		image_data = []
                line_data = []
                hint_data = []
		for ele in datasets:
                    x, y ,z  = self.loadImage(ele[0], ele[1], str(isTrain))             
                    image_data.append(x)                
                    line_data.append(y)
                    hint_data.append(z)     
                    del (x)
                    del (y)
                    del (z)

                del(datasets)
                del (line)
                del(image)         
                return image_data, line_data, hint_data

	    else:
                image_data = []
                line_data = []
                hint_data = []
                line128_data = []
                hint128_data = []
                for ele in datasets:
                    p, q, x, y ,z  = self.loadImage(ele[0], ele[1], str(isTrain))  
                    line128_data.append(p)
                    hint128_data.append(q)
                    image_data.append(x)                
                    line_data.append(y)
                    hint_data.append(z)
                    del (p)
                    del (q)
                    del (x)
                    del (y)
                    del (z)

                del(datasets)
                del (line)
                del(image)
                return line128_data, hint128_data, image_data, line_data, hint_data

        trainDatasets = build_dataSets(image, line, mode, shuffle=False, isTrain=True)
        return trainDatasets

class Datasets_512(Datasets):
    def __init__(self ,batch_size):
        self.batch_size = batch_size
        super().__init__(self, batch_size = self.batch_size)
        
    def get_image_and_line(self):
        image = glob(train_image_datasets_path)
        image.sort()
        line = glob(train_line_datasets_path)
        line.sort()
        return image, line
        
    def _next(self, image, line):   
        return self.buildDataSets(image, line, 'refined')
        
    def _flip(self, image, line, training):
        if training:
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 0)
                line = cv2.flip(line, 0)

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                line = cv2.flip(line, 1)

        return image, line

    def _buildHint(self, image):
        random = np.random.rand
        hint = np.ones_like(image)
        hint += 1
        leak_count = np.random.randint(16, 128)

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

    def loadImage(self, imagePath, linePath, train):
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        line = tf.io.read_file(linePath)
        line = tf.image.decode_jpeg(line, channels=1)
        image = tf.image.resize(image, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        line = tf.image.resize(line, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_128 = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        line_128 = tf.image.resize(line, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image = self.convert2float(image)
        line = self.convert2float(line)
        image_128 = self.convert2float(image_128)
        line_128 = self.convert2float(line_128)

        hint_128 = tf.py_function(self._buildHint,
                                  [image_128],
                                  tf.float32)

        hint_128.set_shape(shape=image_128.shape)
        hint = tf.image.resize(hint_128, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return line_128, hint_128, image, line, hint
