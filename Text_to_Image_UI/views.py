from model.UI import Text_to_Image, RandomCaption
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseServerError
import json
from .models import Survey

def index(request):
    return render(request, 'index.html')

def t2i(request):
    try:
        text = request.POST.get('text')  
        t2i = Text_to_Image()  
        print("Input >>>>> ", text)
        result = t2i.process(text)
    except Exception as e:
        print(e)
        return render(request, '403.html', status='403')

    if result[0]:
        Survey(id=result[1], string=text, url="/static/images/img_{}.jpg".format(result[1])).save()
        context = {
            'num': result[1],
            'rating': 0,
            'survey': True
        }
        return render(request, 't2i.html', context)

def rate(request):
    try:
        id = request.POST.get('id')
        rating = request.POST.get('rating', 0)
        Survey.objects.filter(id=id).update(rating=rating)
        return HttpResponse('true')
    except:
        return HttpResponseServerError('false')

def random(request):
    random = RandomCaption()
    return HttpResponse(random.get())