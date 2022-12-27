# Seismic-bumps data set

El archivo .ipynb contiene las instrucciones para entrenar y testear 3 modelos lineales/cuadráticos y 3 modelos no lineales sobre el dataset seismic-bumps. Los modelos elegidos para el estudio son regresión logística, naive bayes y K nearest neighbours; mientras que los elegidos para los modelos no lineales son multi-layer perceptron, gradient boosting y random forest. Una vez entrenados y testeados, se comprueban los resultados obtenidos, la precisión y la puntuación de ellos para ver cuál de los seis obtiene mejores predicciones. 

## Descarga del dataset e instalación de paquetes

1) Dataset

Para descargar el dataset diríjase al siguiente [enlace](http://archive.ics.uci.edu/ml/datasets/seismic-bumps). Enconrtará dos enlaces llamados Data Folder y Data Set Description. En Data folder encontrará el data set llamado seismic-bumps.arff. Descárguelo y agréguelo a este mismo directorio.

Como se encuentra en otro formato diferente al habitual, puede convertir archivos .arff a .csv con el archivo python adjunto. Ejecute el siguiente comando:

```bash
python3 arffToCsv.py
```

Este comando convierte todos los archivos .arff en .csv. Una vez obtenido el dataset con el formato apropiado, puede continuar con la instalación de paquetes.

En caso de que alguno de los pasos anteriores haya fallado, puede encontrar todos los ficheros correctamente configurados y preparados para la ejecución en el siguiente [repositorio de github](https://github.com/AlphaIcarus/PracticaAPA.git)

Puede descargar el repositorio con el siguiente comando:

```bash
git clone https://github.com/AlphaIcarus/PracticaAPA.git
```

2) Paquetes utilizados

La lista de paquetes utilizados se encuentra en el fichero requirements.txt. Estos paquetes se pueden descargar utilizando los siguientes comandos:

```bash
pip install foobar
```

## Ejecución del notebook

Para visualizar el notebook debe hacer doble click sobre el archivo .ipynb para abrir la máquina virtual. Alternativamente, puede abrir una terminal en el directorio donde se encuentra el notebook, y ejecutar el comando:

```bash
jupyter notebook
```

Este le llevará al contenido del directorio desde el menú del notebook, y allí puede seleccionar el archivo .ipynb.

El Jupyter Notebook ya mantiene la ejecución previa, por lo que no hace falta ser ejecutado de nuevo. Si se desea ejecutar de nuevo, debe presionar Kernel > Restart & Run all.

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
