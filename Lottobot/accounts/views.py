from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import SignupSerializer





class SignupAPIView(APIView):
    permission_classes = [AllowAny]
    print("---"*18)
    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        print("---"*2)
        if serializer.is_valid(raise_exception=True):
            user = serializer.save()
            print("---"*3)
            return Response(
                {
                    "msg": "signup 성공",
                    "ID": user.username,
                },
                status=status.HTTP_201_CREATED
            )



class LogoutAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        refresh_token = request.data.get("refresh")
        if refresh_token:
            try:
                token = RefreshToken(refresh_token)
                token.blacklist()
            except Exception as e:
                return Response(
                    {
                        "message": "로그아웃 실패",
                        "error": str(e),
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        return Response(
            {
                "message": "로그아웃 완료!",
            },
            status=status.HTTP_204_NO_CONTENT
        )


class DeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def delete(self, request):
        request.user.delete()
        return Response(
            {
                "msg": "회원탈퇴 완료"
            },
            status=status.HTTP_204_NO_CONTENT
        )



