from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Instanciamos un objeto de la clase fastapi para construir la aplicación
app = FastAPI(title='STEAM Games: Consultas', description='Esta aplicación permite realizar consultas sobre videojuegos, reseñas de usuarios, recomendaciones y más')

# Cargamos las tablas limpias en .parquet
itemsxgames = pd.read_parquet('itemsxgames.parquet')
reviewsxgames = pd.read_parquet('reviewsxgames.parquet')
steamxgames = pd.read_parquet('steamxgamesml.parquet')
funcion_4 = pd.read_parquet(r'data/user_reviews.parquet')
merged_df = pd.merge(funcion_4, steamxgames, on='item_id', how='inner')
merged_df

# ruta inicial
@app.get("/")
async def index():
    mensaje = 'Bienvenidos a mi API , encontraras algunas consultas y recomendaciones sobre juegos de STEAM'
    return {'Mensaje': mensaje}

#Funcion1
@app.get("/Play-Time-Genre/{genero}", name="Tiempo de juego por género")
def PlayTimeGenre(genero: str):
    '''
    Debe devolver año con mas horas jugadas para dicho género.
    Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
    
    '''
    # Filtramos el dataframe 'items_games' respecto al parámetro genero
    df = itemsxgames[itemsxgames['genres']== genero]
    
    # Agrupamos el dataframe anterior por año de lanzamiento, suma de minutos de juego y ordenamos en forma descendente
    agrupado = df.groupby('year_release')['playtime_forever'].sum().sort_values(ascending=False)

    # El valor máximo tendrá índice [0]
    anio = agrupado.index[0]
    
    return {f"Año de lanzamiento con más horas jugadas para el género {genero}": int(anio)}

#Funcion2
@app.get("/User-For-Genre/{genero}", name="Usuario con mas minutos jugados para un género")
def UserForGenre(genero: str):
    '''
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
    Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
    '''
    # Filtramos el dataframe 'itemsxgames' respecto al parámetro genero
    df = itemsxgames[itemsxgames['genres'] == genero]

    # Agrupamos el dataframe anterior por usuario, suma de horas de juego y ordenamos en forma descendente
    agrupado = df.groupby('user_id')['playtime_forever'].sum().sort_values(ascending=False)

    # El valor máximo tendrá índice [0]
    user = agrupado.index[0]

    # Tomamos las filas del dataframe util que contengan su respectivo usuario (user)
    df_genero_user = df[df['user_id'] == user]

    # Agrupamos respecto a los años y suma de horas de juego
    horas_jugadas = round(df_genero_user.groupby('year_release')['playtime_forever'].sum() / 60, 3)

    # Guardamos la serie 'horas_jugadas' en una lista
    lista_horas_jugadas = [{'Año': int(anio), 'Horas': horas} for anio, horas in horas_jugadas.items()]

    return {f"Usuario con más horas jugadas para género {genero}": user, "Horas jugadas": lista_horas_jugadas}

#Funcion3
@app.get("/User-Recommend/{anio}", name="Top 3 de juegos MÁS recomendados por usuarios por año")
def UserRecommend(anio: int):
    '''
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
    Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

    '''
    # Si el año de lanzamiento(year_release) no coincide con alguno de los años en los que se hace una reseña(year_posted), se retorna un mensaje de erro
    if anio not in reviewsxgames['year_posted'].unique():
        return f"Año fuera de rango, ingrese un año válido"
    
    else:
        # Filtramos el dataframe con las filas cuyo año de posteo(year_posted) es mayor o igual al año de publicación(year_release)
        df = reviewsxgames[reviewsxgames['year_posted']>=reviewsxgames['year_release']]
        
        # Filtramos el dataframe 'df' para el año parámetro y la columna sentiment_analysis sea positivo(2) o neutro(1)
        df_anio_dado = df[(df['year_posted']==anio) & (df['sentiment_analysis'].isin([1,2]))]

        # Agrupamos el dataframe 'df_anio_dado' por título del juego ('title'), sumamos las recomendaciones('recommend') para tener los juegos más recomendados y ordenamos de forma descendente
        top = df_anio_dado.groupby('title')['recommend'].sum().sort_values(ascending=False)

        # Construimos el top3
        top3 = [{"Puesto 1": top.index[0]}, {"Puesto 2": top.index[1]}, {"Puesto 3": top.index[2]}]

    return top3

#Funcion4
@app.get("/Users-Worst-Developer/{anio}", name="Top 3 de desarrolladores con juegos MENOS recomendados por usuario")
def UsersWorstDeveloper(año: int):
    '''
    Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
    Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    '''
    # Filtra el DataFrame para el año dado
    filtered_df = merged_df[merged_df['year_posted'] == año]

    # Filtra juegos menos recomendados (recommend=0 y sentiment_analysis<2)
    less_recommended = filtered_df[(filtered_df['recommend'] == 0) & (filtered_df['sentiment_analysis'] < 2)]

    # Agrupa por desarrolladora y cuenta la cantidad de juegos menos recomendados
    developer_counts = less_recommended.groupby('developer').size().reset_index(name='count')

    # Ordena en orden ascendente por la cantidad de juegos menos recomendados
    sorted_developers = developer_counts.sort_values(by='count')

    # Toma las 3 desarrolladoras con menos juegos recomendados
    top_3 = sorted_developers.head(3)

    # Crea el resultado en el formato solicitado
    result = [{"Puesto {}: {}".format(i+1, row['developer'])} for i, (index, row) in enumerate(top_3.iterrows())]

    return result

#Funcion5
@app.get('/Sentiment-analysis/{anio}', name='lista con la cantidad de registros de reseñas de usuarios')
def sentiment_analysis(desarrolladora: str):
    '''
    Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.
    Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}
    '''
    # Filtra el DataFrame para la desarrolladora dada
    filtered_df = merged_df[merged_df['developer'] == desarrolladora]

    # Cuenta la cantidad de registros por análisis de sentimiento
    sentiment_counts = filtered_df['sentiment_analysis'].value_counts().sort_index()

    # Mapea los valores numéricos de sentiment_analysis a etiquetas
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_counts.rename(index=sentiment_labels, inplace=True)

    # Crea el resultado en el formato solicitado
    result = {desarrolladora: sentiment_counts.to_dict()}

    return result
    
#MODELO ML
# Creamos una instancia de la clase CountVectorizer
vector = CountVectorizer(tokenizer=lambda x: x.split(', '))

# Dividimos cada cadena de descripción en palabras individuales y creamos una matriz de conteo 'matriz_generos' que representa cuántas veces aparece cada género en cada videojuego.
matriz_generos = vector.fit_transform(steamxgames['description'])

@app.get('/Juegos-recomendados/{id_producto}', name='lista con juegos recomendados por juego ingresado')


def recomendacion_juego(id_producto: int):
    '''
    Se ingresa el id de producto (item_id) y retorna una lista con 5 juegos recomendados similares al ingresado (title).
    '''
    # Si el id ingresado no se encuentra en la columna de id de la tabla 'steam_games', se le pide al usuario que intente con otro id
    if id_producto not in steamxgames['item_id'].values:
        return 'El ID no existe, intente con otro'
    else:
        # Buscamos el índice del id ingresado
        index = steamxgames.index[steamxgames['item_id'] == id_producto][0]

        # De la matriz de conteo, tomamos el array de géneros con índice igual a 'index'
        generos_index = matriz_generos[index]

        # Calculamos la similitud coseno entre los géneros de entrada y los géneros de las demás filas: cosine_similarity(generos_index, matriz_generos)
        # Obtenemos los índices de las mayores similitudes mediante el método argsort() y las similitudes ordenadas de manera descendente
        # Tomamos los índices del 1 al 6 [0, 1:6] ya que el índice 0 es el mismo índice de entrada
        indices_maximos = np.argsort(-cosine_similarity(generos_index, matriz_generos))[0, 1:6]

        # Construimos la lista de recomendaciones
        recomendaciones = []
        for i in indices_maximos:
            recomendaciones.append(steamxgames['title'][i])

        return recomendaciones

