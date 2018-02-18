import conjuntoDeDatos as data

def probar():
    data = conjuntoDeDatos.leerDatosDeEntrenamiento(rutaDeDatosDeEntrenamiento, tamanoDeImagenes, clases, validation_size=tamanoDeDataDeValidacion)
    print