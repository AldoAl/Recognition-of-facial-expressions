import skimage
import skimage.io
import skimage.transform
import os
import random

#Retornamos una lista con los nombres de los archivos/files que se encuentren en "path"
def list_files_of_directory(path):
    list_files = []
    for filename in os.listdir(path):
        list_files.append(filename)
    return list_files

# Sacamos el nombre de la clase del archivo
def get_nameCls(file):
    namefile = file.split(".")
    # assert es como un if, pero si no se cumple la condicion
    #  el programa se detiene y muestra el mensaje
    assert len(namefile) > 2, 'El archivo no tiene extencion'
    # anadimos el nombre de la clase solo si no se encuentra en la lista list_nameCls
    # ASUMIMOS que el nombre de la clase se encuentra en el penultimo lugar, ejm: subject09.surprised.jpg
    nameCls = namefile[len(namefile) - 2]
    return nameCls

# Obtenemos la lista de clases segun la lista de archivos generada por list_files_of_directory()
def get_nameClasses(list_files):
    list_nameCls = []
    for i in range(len(list_files)):
        nameCls = get_nameCls(list_files[i])
        if nameCls not in list_nameCls:
            list_nameCls.append(nameCls)
    return list_nameCls

# Mostramos el contenido que posee cualquier lista
def showList(lista):
    cont = 0
    for i in lista:
        print(cont,i)
        cont += 1

# Cargamos la imagen(imread) y recortamos(crop) la imagen para que sea cuadrada
# y hacemos un resize con dim_image dado
def resize_image(image, dim_image=64, scale=255):
    # load image
    img = load_image(image)
    print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to (dim_image, dim_image)
    resized_img = skimage.transform.resize(crop_img, (dim_image, dim_image), mode='constant')
    return resized_img

# Cargamos laimagen, el resultado es la imagen en una matriz
# la matriz de la imagen por defecto esta en una escala [0 255]
def load_image(image):
    return skimage.io.imread(image)

# Solo es guardar la imagen resultante de la funcion resize_image
def save_image(image_name,extencion,image_skimage):
    skimage.io.imsave(image_name + "." + extencion, image_skimage)

# Nos aseguramos que el directorio exista, si no existe , lo creamos
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# si la carpeta donde estara la dataset preprocesadoa (dataset_new) no existe lo creamos
# luego, hacemos un resize_image() y guardamos la nueva imagen con extencion jpg en la carpeta dataset_new
def create_dataset_new(dataset_original, dataset_new,dim_image=64):
    if not os.path.exists(dataset_new):
        ensure_dir(dataset_new)
        list_files = list_files_of_directory(dataset_original)
        for file in list_files:
            img = resize_image(dataset_original + file, dim_image)
            save_image(dataset_new + file,"jpg", img)

# Retorna la posicion en la que se encuentra nameCLs con respecto a list_nameCls
def get_numClass(nameCls,list_nameCls):
    for i in range(len(list_nameCls)):
        if nameCls == list_nameCls[i]:
            return i

# Leo las imagenes y las coloco en una matriz
# las imagenes estan en una esca [0 255], las convertire a escala [0 1], y las convertirea a vector 64*64=4096
# La matriz tendra una dimension de (165,4096),
def load_dataset_all(path_dataset_new,list_files,list_nameCls):
    data_all = []
    data_label_all = []
    for file in list_files:
        image = load_image(path_dataset_new + file)
        # normalizo la imagen de escal [0 255] a [0 1],
        # NOTA: por defecto es 255, pero se puede sacar el maximo de la matriz image y dividirlo con eso
        image = image/255
        # convierto la matriz en array con la funcion ravel() (64,64) = (4096,)
        image = image.ravel()
        data_all.append(image)
        #Guardamos el label
        nameCls = get_nameCls(file)
        data_label_all.append(get_numClass(nameCls,list_nameCls))
    return (data_all,data_label_all)

# Generamos indices aleatorio segun el tamano "n" que queramos
# Los indices aleatorios son termutaciones de numeros sin repetirse, ejm: [0,1,2,3] => [2,0,3,1]
def crear_indices_aleatorios(n):
    # crea un vector de 0 a n
    indices = range(n)
    # genera los indices de forma desordenada
    random.shuffle(indices)
    return indices

def create_data_train_test(indices_aleatorios,data_all,data_label_all,test_size=0.25):
    data_train = []
    data_test = []
    label_train = []
    label_test =[]
    tam_test_size = (int)(len(data_label_all)*test_size)
    cont = 0
    for i in indices_aleatorios:
        # Llenamos la parte del test
        if cont < tam_test_size:
            data_test.append(data_all[i])
            label_test.append(data_label_all[i])
        else: # Llenamos la parte del train
            data_train.append(data_all[i])
            label_train.append(data_label_all[i])
        cont += 1
    return (data_train,label_train,data_test,label_test)

# --------------------------------Main en python ---------------------------------------
if __name__=="__main__":
    path_dataset_original = "yalefaces/"
    path_dataset_new = "yalefaces_new/"

    create_dataset_new(path_dataset_original,path_dataset_new, 64)

    list_files = list_files_of_directory(path_dataset_new)
    list_nameCls = get_nameClasses(list_files)
    showList(list_nameCls)

    data_all,data_label_all = load_dataset_all(path_dataset_new,list_files,list_nameCls)
    print("data_all:",len(data_all),len(data_all[0]))
    print("data_label_all:",len(data_label_all))

    indices_aleatorios = crear_indices_aleatorios(len(data_label_all))
    data_train,label_train,data_test,label_test = create_data_train_test(indices_aleatorios,data_all,data_label_all)
    print("data_train:", len(data_train), len(data_train[0]))
    print("label_train:", len(label_train))
    print("data_test:", len(data_test), len(data_test[0]))
    print("label_test:", len(label_test))

    print("Finish!")
