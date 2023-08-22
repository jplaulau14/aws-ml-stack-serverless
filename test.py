import boto3

client = boto3.client('sagemaker-runtime')

endpoint_name = "blazingtext-2023-08-21-13-54-54-562"                                     
content_type = "application/json"                                  
payload = '{"instances": ["A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery."]}'                                            
response = client.invoke_endpoint(
    EndpointName=endpoint_name, 
    ContentType=content_type,
    Body=payload
    )

print(response['Body'].read().decode('utf-8'))           
                                      