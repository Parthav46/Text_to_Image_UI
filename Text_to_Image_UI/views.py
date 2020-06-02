from model.UI import Text_to_Image
from django.shortcuts import render
import json
# from django.views.decorators.csrf import csrf_exempt

def index(request):
    return render(request, 'index.html')

def t2i(request):
    t2i = Text_to_Image()
    text = request.POST.get('text')
    print("Input >>>>> ", text)
    context = {
        'num': t2i.process(text)[1]
    }
    return render(request, 't2i.html', context)
