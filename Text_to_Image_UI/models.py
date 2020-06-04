from django.db import models

class Survey(models.Model):
	id = models.IntegerField(primary_key=True)
	string = models.CharField(max_length=128)
	url = models.URLField()
	rating = models.IntegerField(blank=True, null=True)
