import conjuntoDeDatos as data


rutaDeDatos='imagenes'
clases = ['dogs','cats']
numeroClases = len(clases)
tamanoDeDataDeValidacion = 0.2

def probar():
    data = conjuntoDeDatos.leerDatosDeEntrenamiento(rutaDeDatosDeEntrenamiento, 
                tamanoDeImagenes, 
                clases, 
                validation_size=tamanoDeDataDeValidacion)

def cargarDatosDeEntrenamiento(rutaDeDatos, tamanoDeImagenes, clases):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in clases:
        index = clases.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (tamanoDeImagenes, tamanoDeImagenes),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(clases))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls    

cargarDatosDeEntrenamiento(rutaDeDatos, tamanoDeImagenes, clases)    