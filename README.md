# Proyecto MLOps: Sistema de Recomendación de Videojuegos para Steam

¡Bienvenido al proyecto de Machine Learning Operations (MLOps) donde abordaremos el desarrollo de un sistema de recomendación de videojuegos para la plataforma Steam! En este proyecto, asumiré el papel de un MLOps Engineer en Steam, enfrentándome al desafío de llevar un modelo de recomendación.

## Desarrollo del Proyecto

### 1. Data Engineering

#### Transformaciones de Datos:
- **Limpieza del Dataset:** Inicialmente, me encontré con datos en bruto y poco maduros. Realicé tareas de limpieza, eliminación de columnas innecesarias y ajuste de formato para garantizar la calidad de los datos.
- **Formato de Datos para Consumo:** Aseguré que el conjunto de datos sea legible y eficiente para su consumo, preparando el terreno para el desarrollo del modelo y la API.

#### Feature Engineering:
- **Análisis de Sentimiento (NLP):** En el dataset `user_reviews`, apliqué análisis de sentimiento a las reseñas de los juegos. Creé la columna 'sentiment_analysis' con valores 0 (malo), 1 (neutral) y 2 (positivo). Esto facilitará el trabajo de los modelos de machine learning y el análisis de datos.

### 2. Desarrollo de la API

#### FastAPI y Endpoints:
- Use FastAPI para exponer los datos de la empresa a través de una API RESTful.
- Desarrollé las siguientes funciones y endpoints:
  - `PlayTimeGenre(genero: str):` Devuelve el año con más horas jugadas para el género dado.
  - `UserForGenre(genero: str):` Devuelve el usuario con más horas jugadas para el género dado y una lista de acumulación de horas jugadas por año.
  - `UsersRecommend(año: int):` Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
  - `UsersWorstDeveloper(año: int):` Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado.
  - `sentiment_analysis(empresa_desarrolladora: str):` Devuelve un diccionario con el análisis de sentimiento para la empresa desarrolladora especificada.

#### Consumo de la API:
- Garanticé que la API sea consumible desde cualquier dispositivo conectado a internet, siguiendo los principios de RESTful.

### 3. Deployment

#### Plataforma de Deployment:
- Opté por Render para el deployment de la API, aprovechando un tutorial que facilitó el proceso.
- Consideré otras opciones como Railway para asegurar la accesibilidad de la API desde la web.

### 4. Análisis Exploratorio de Datos (EDA)

#### Exploración Manual:
- Realicé un análisis exploratorio de datos manual sin el uso de librerías automatizadas.
- Investigé relaciones entre variables, identifiqué outliers y anomalías, y busqué patrones interesantes en los datos.

#### Nubes de Palabras:
- Utilicé nubes de palabras para visualizar las palabras más frecuentes en los títulos de los juegos, con la intención de mejorar el sistema de predicción.

### 5. Modelo de Aprendizaje Automático

#### Sistema de Recomendación:
- Opté por desarrollar un sistema de recomendación user-item utilizando similitud del coseno.
- Implementé el endpoint `recomendacion_usuario(id_de_usuario)` en la API para proporcionar una lista de 5 juegos recomendados para un usuario dado.

### 6. Links:
- API DEPLOYADA: https://pi-mlops-7yi5.onrender.com/docs
- VIDEO PRESENTACIÓN: [Ver video en YouTube](https://youtu.be/j8PKHDoxH8E)




