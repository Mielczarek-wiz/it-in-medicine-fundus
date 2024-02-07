from tools import *

	
def get_y(expert_mask, cell_size, shift):
    img_width = expert_mask.shape[0]
    img_height = expert_mask.shape[1]
    begin_from = int(cell_size/2)
    pixels = []
    
    for x in range(begin_from, img_width-begin_from, shift):
        for y in range(begin_from, img_height-begin_from, shift):
            if(expert_mask[x][y]>100): 
                pixels.append(1)
            else:
                pixels.append(0)
    return pixels
	
	
def get_x(input_bitmap, cell_size, shift):
    img_width = int((input_bitmap.shape[0]) / cell_size) * cell_size
    img_height = int((input_bitmap.shape[1])/ cell_size) * cell_size
    
    if(shift==1):
        img_width = input_bitmap.shape[0]
        img_height = input_bitmap.shape[1]
        top, bottom, left, right = [int(cell_size/2)]*4
        input_bitmap = cv2.copyMakeBorder(input_bitmap, top, bottom,
                                        left, right, cv2.BORDER_CONSTANT)
        
    parameters = []
    for x in range(0, img_width, shift):
        for y in range(0, img_height, shift):
                
            cell = input_bitmap[x : x+cell_size, y : y+cell_size]

            green_var = np.var(cell[:,:,1])
            
            cell_gray = rgb2gray(cell)
            moments = cv2.moments(cell_gray)
            moments_list = list(moments.values())
            hu_moments = cv2.HuMoments(moments)
            hu_moments_2 =[]
            for i in range(len(hu_moments)):
                hu_moments_2.append(hu_moments[i][0])
            parameters.append([green_var, *moments_list, *hu_moments_2])
            
    return parameters
	
	
def KNN_learn(x_train, y_train):
    knn_model = KNeighborsClassifier(n_neighbors=13, n_jobs=-1)
    knn_model.fit(x_train, y_train)
    knnPickle = open('knnpickle_file', 'wb')
    pickle.dump(knn_model, knnPickle)
	
	
def KNN_predict(x_test, y_test, img_width, img_height):
    knn_model = pickle.load(open('knnpickle_file', 'rb'))
    x_predict=knn_model.predict(x_test)

    x_predict = np.array(x_predict)
    shape = (img_width, img_height)
    mask = x_predict.reshape(shape)
    plt.figure()
    io.imshow(mask, cmap='gray')

    statistical_analysis(mask, y_test)
	
def learn(path_original='train/image/*.jpg', path_label='train/label/*.tif', cell_size=5, train_limit=2, scale=0.1):
    x_train = []
    y_train = []
    shift = cell_size
    
    images = io.ImageCollection(path_original)
    tifes = io.ImageCollection(path_label)
    lista = []
    for i in range(len(images)):
        lista.append((images[i], tifes[i]))
    random.shuffle(lista)
    
    for image_number, (image, tiff) in enumerate(lista):
        image=RescaleImage(image, scale)
        tiff=RescaleImage(tiff, scale)
        xs = get_x(image, cell_size, shift)
        ys = get_y(tiff, cell_size, shift)
        for x in xs:
            x_train.append(x)
        for y in ys:
            y_train.append(y)
        if(image_number == train_limit):
            break
            
    sampler = RandomUnderSampler(sampling_strategy=1, random_state=200)
    x_train, y_train = sampler.fit_resample(x_train, y_train)
    
    KNN_learn(x_train, y_train)
	
def predict(path_test='test/image/*.jpg', path_label='test/label/*.tif', how_many=2, cell_size=5, scale=0.1):
    test_bitmaps = io.ImageCollection(path_test)
    expert_masks = io.ImageCollection(path_label, as_gray=True)

    for i in range(how_many):
        test=RescaleImage(test_bitmaps[i], scale)
        x_test = get_x(test, cell_size, 1)
        y_test = RescaleImage(expert_masks[i],scale)
        plt.figure()
        io.imshow(test)
        plt.figure()
        io.imshow(y_test, cmap='gray')
        
        y_test[y_test<=100] = 0
        y_test[y_test>0] = 1

        start = time.time()
        KNN_predict(x_test, y_test, test.shape[0], test.shape[1])
        end = time.time()
        print(f"time: {end - start} s")