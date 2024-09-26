from django.shortcuts import render
from movie.models import Movie
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import json
from dotenv import load_dotenv, find_dotenv



def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommendation(request):
    req = request.GET.get('searchMovie')

    if req:
#Se lee del archivo .env la api key de openai
        _ = load_dotenv('../api_keys.env')
        client = OpenAI( api_key=os.environ.get('opeanai_api_key'),)
        items = Movie.objects.all()
        # Obtener el embedding de la consulta
        emb_req = get_embedding(req, client)
        
        # Calcular la similitud
        sim = []
        for item in items:
            emb = list(np.frombuffer(item.emb)) 
            sim.append(cosine_similarity(emb, emb_req))
        
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        recommended_movie = items[idx]

        return render(request, 'recommendations.html', {'movie': recommended_movie, 'search_term': req})

    else:
        return render(request, 'recommendations.html')
# Create your views here.
