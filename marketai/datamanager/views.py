import js2py as parser
from django.db import connection
from django.http import JsonResponse
from rest_framework.decorators import api_view

@api_view(["POST"])
def datamanager(request):
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM datamanager_datamanager")
        rows = cursor.fetchall()

    return JsonResponse(
        rows, safe = False
    )

@api_view(["POST"])
def download(request):
    eval_fetchdata, fetchdata = parser.run_file("datamanager\\lib\\fetchdata.js")
    data = fetchdata.fetchMarket()
    