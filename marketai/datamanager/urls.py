from . import views
from django.urls import path

urlpatterns = [
    path('', views.datamanager, name='datamanager'),
    path('download', views.datamanager, name='download'),
]