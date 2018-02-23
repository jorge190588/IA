import tensorflow as tf

class CNN(object):
  """ contructor de la clase """
  def __init__(self):
        
    def crearCapaAplanada(capaNoAplanada):
        #We know that the shape of the layer will be [batch_size img_size img_size num_channels]
        #sabemos que la forma de una capa podria ser [tamanoPaquete tamanoImagene tamanoImagen numeroCanales]

        # pero conseguiremos esto desde una capa previa
        # But let's get it from the previous layer.
        formaDeLaCapa = capaNoAplanada.get_shape()

        ## El numero de caracteristica podria ser altoImagen * anchoImagen * numeroCanales.  
        ## Pero nosotros deberiamos calcularlo esto en lugar de codificarlo con a la fuerza o forzado.
        numeroDeCaracteristicas = formaDeLaCapa[1:4].num_elements()

        ## ahora, aplanaremos la capa, asi que tendremos que remodelar (reshape) la cantidad de caracteristicas (num_features)
        capaAplanada = tf.reshape(capaNoAplanada, [-1, numeroDeCaracteristicas])

        #retornar la capa aplanda.
        return capaAplanada

    def crearCapaTotalmenteConectada(tensorDeEntrada,num_inputs,num_outputs,use_relu=True):
        #vamos a definir los pesos y sesgos entrenables
        pesos = crearPesos(shape=[num_inputs, num_outputs])
        sesgos = crearSesgos(num_outputs)

        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        capaTotalmenteConectada = tf.matmul(tensorDeEntrada, pesos) + sesgos
        if use_relu:
            capaTotalmenteConectada = tf.nn.relu(capaTotalmenteConectada)

        return capaTotalmenteConectada

    # crear pesos
    def crearPesos(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    #crear sesgos
    def crearSesgos(size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    #funcion para crear un capa convolucional.
    def crearCapaConvolucional(tensorDeEntrada,numeroDeCanales, tamanoDelFiltro,numeroDeFiltros):

        ## We shall define the weights that will be trained using create_weights function.
        ## Debemos definir los pesos que seran entrenables usango la funcion crearPesos
        pesos = crearPesos(shape=[tamanoDelFiltro, tamanoDelFiltro, numeroDeCanales, numeroDeFiltros])

        ## We create biases using the create_biases function. These are also trained.
        ## creamos los sesgos usanto la funcion crearSesgos. Estos son tambien entrenados
        sesgos = crearSesgos(numeroDeFiltros)

        ## Creando la capa convolucional
        capa = tf.nn.conv2d(input=tensorDeEntrada,filter=pesos,strides=[1, 1, 1, 1],padding='SAME')

        #Agregar los sesgos a la capa.
        capa += sesgos

        ## We shall be using max-pooling. see more information on https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
        # Estaremos usando el metodo cnn.max-pooling de tensorflow. El algoritmo realiza el maximo agrupamiento en la entrada
        ##Los parametros de cnn.max_pool son:
        # value =  Un tendor de 4 dimension con formato especificado por data_format.
        # ksize =  Un tensor de tipo entero (int) de 1 dimension con 4 elementos.  
        ##         Representan el tama;o de la ventana para cada dimension del tensor de entrada.
        # strides= Strides significa pasos. Es un tensor de tipo entero (int) de 1 dimension con 4 elementos.
        ##         El paso de la ventana deslizante para cada dimension del tensor de entrada.
        # padding= Es el relleno.  Es un texto, ya sea SAME o VALID.   Existe un algoritmo para el relleno (padding)
        # ver documentacion de tf.cnn.max_pooling
        capa = tf.nn.max_pool(value=capa,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')

        ## Output of pooling is fed to Relu which is the activation function for us.
        capa = tf.nn.relu(capa)

        # retornar un tensor con el formato especificado en data_format.  Es un tensor resultante con el maximo de agrupamiento.
        return capa
    