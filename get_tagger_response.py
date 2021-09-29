import requests

def get_tagger_response(text):
    url = "https://api.parlamento2030.es/tagger/"   
    resp = requests.post(url, data = {'text': text})    
    return(resp.text)

result = get_tagger_response("Libros de texto")