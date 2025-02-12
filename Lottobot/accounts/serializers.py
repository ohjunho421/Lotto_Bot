from rest_framework import serializers
from .models import User



class SignupSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = User
        fields = ["username", "password"]
    
    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            password=validated_data["password"]
        )
        user.save()
        return user
    
