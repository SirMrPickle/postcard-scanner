import boto3
import botocore
import json
import base64

# Create LLM clients
textractClient = boto3.client("textract")
bedrockClient = boto3.client("bedrock-runtime")


def processPostcard(postcard):
    # 0: Read the image.
    with open(postcard, "rb") as imageFile:
        imageBytes = (
            imageFile.read()
        )  # You can"t just shove a raw image into this LLM, it needs to be in bytes.

    """ Detects the same thing, no need to make 2 of the same calls. I'll just send it to claude to differenciate.
    # 1: Printed text detection
    printedTextResponse = textractClient.detect_document_text(
        Document={"Bytes": imageBytes}
    )

    # 1.5. Parse returned text:
    printedText = []
    for block in printedTextResponse["Blocks"]:
        if block["BlockType"] == "LINE":
            printedText.append(block["Text"])
    """

    # Numero Dos: Handwritten text detection:
    writtenTextResponse = textractClient.analyze_document(
        Document={"Bytes": imageBytes}, FeatureTypes=["FORMS"]
    )

    writtenText = [
        block["Text"]
        for block in writtenTextResponse["Blocks"]
        if block["BlockType"] == "LINE"
    ]

    # 3. Smart analysis feat. Bedrock Claude 3
    # smartAnalysis = bedrockAnalysis(imageBytes, printedText, writtenText)
    smartAnalysis = bedrockAnalysis(imageBytes, writtenText)

    return {
        "postcard_path": postcard,
        # "printed_text": printedText,
        "handwritten_text": writtenText,
        "claude_analysis": smartAnalysis,
    }


# def bedrockAnalysis(imageBytes, printedText, handwrittenText):
def bedrockAnalysis(imageBytes, handwrittenText):
    # Encode image for Bedrock
    encodedImage = base64.b64encode(imageBytes).decode("utf-8")

    # Create the prompt: "You are a blowfish."
    claudePayload = f"""
        You are a helpful assistant trained to analyze the text found on the back of a vintage postcard. Below is a list of text lines detected via OCR.

        Your tasks:
        1. Sort each line into either:
        - "printed_text": professional or machine-printed lines (publisher info, labels like "Postcard", etc.)
        - "handwritten_text": personal messages, addresses, or anything likely written by hand.
        2. Provide a brief analysis of the postcard's content and tone based on the handwritten portion (if present).

        Input lines:
        {json.dumps(handwrittenText, indent=2)}

        Return your response as a JSON object with the following format:
        {{
            "title": "",
            "subjects": [],
            "photographer": "",
            "publisher": "",
            "series_or_collection": "",
            "image_description": "",
            "image_copyright_date": "",
            "location_depicted": "",
            "street_address_location": "",
            "card_manufacturing_location": "",
            "time_period_estimated": "",
            "text_present_front": "",
            "text_present_back": "",
            "languages_present": [],
            "handwritten_markings": "",
            "visible_codes_or_catalog_numbers": "",
            "printed_in": "",
            "image_style": "",
            "condition_notes": "",
            "historical_or_cultural_context": "",
            "stamp_price": "",
            "additional_notes": ""
        }}
    """

    # Define request sent to endpoint
    requestBody = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encodedImage,
                        },
                    },
                    {"type": "text", "text": claudePayload},
                ],
            }
        ],
    }

    # Actually try and send the thing
    try:
        response = bedrockClient.invoke_model(
            modelId="us.anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(claudePayload),
        )

        responseBody = json.loads(response["body"].read())
        return responseBody["content"][0]["text"]

    # You know that this is.
    except Exception as e:
        print(f"Bedrock error: {e}")
        return None


print(json.dumps(processPostcard("output/final/card0001.png"), indent=2))
