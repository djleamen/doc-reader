"""
Serializers for the RAG App API.
"""

from rest_framework import serializers
class QueryRequestSerializer(serializers.Serializer):
    '''Serializer for query requests.'''
    question = serializers.CharField(max_length=5000, required=True)
    index_name = serializers.CharField(max_length=255, default='default')
    k = serializers.IntegerField(min_value=1, max_value=20, required=False)
    include_sources = serializers.BooleanField(default=True)
    include_scores = serializers.BooleanField(default=True)

    def create(self, validated_data):
        '''Not implemented - this is a request serializer.'''
        raise NotImplementedError

    def update(self, instance, validated_data):
        '''Not implemented - this is a request serializer.'''
        raise NotImplementedError


class DocumentUploadSerializer(serializers.Serializer):
    '''Serializer for document upload requests.'''
    files = serializers.ListField(
        child=serializers.FileField(),
        required=True,
        allow_empty=False
    )
    index_name = serializers.CharField(max_length=255, default='default')

    def create(self, validated_data):
        '''Not implemented - this is a request serializer.'''
        raise NotImplementedError

    def update(self, instance, validated_data):
        '''Not implemented - this is a request serializer.'''
        raise NotImplementedError


class DocumentSerializer(serializers.Serializer):
    '''Serializer for document information.'''
    filename = serializers.CharField()
    content = serializers.CharField()
    metadata = serializers.DictField()

    def create(self, validated_data):
        '''Not implemented - this is a response serializer.'''
        raise NotImplementedError

    def update(self, instance, validated_data):
        '''Not implemented - this is a response serializer.'''
        raise NotImplementedError


class QueryResponseSerializer(serializers.Serializer):
    '''Serializer for query responses.'''
    query = serializers.CharField()
    answer = serializers.CharField()
    source_documents = serializers.ListField(
        child=DocumentSerializer(),
        required=False
    )

    def create(self, validated_data):
        '''Not implemented - this is a response serializer.'''
        raise NotImplementedError

    def update(self, instance, validated_data):
        '''Not implemented - this is a response serializer.'''
        raise NotImplementedError
    confidence_scores = serializers.ListField(
        child=serializers.FloatField(),
        required=False
    )
    metadata = serializers.DictField()
