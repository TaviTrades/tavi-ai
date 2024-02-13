# from rest_framework.views import APIView
# from rest_framework.response import Response
# from .service import get_ai_response

# class AIChatView(APIView):
#     def post(self, request, *args, **kwargs):
#         user_input = request.data.get("input")
#         response = get_ai_response(user_input)
#         return Response({"response": response})


from rest_framework.views import APIView
from rest_framework.response import Response
from .service import get_ai_response

class AIChatView(APIView):
    def post(self, request, *args, **kwargs):
        user_input = request.data.get("input")
        response = get_ai_response(user_input)
        return Response({"response": response})
