from blip_image_caption_large import Blip_Image_Caption_Large

# Test the local image caption pipeline with wikipedia image 
def test_blip_image_caption_local_model():
    image_caption_model = Blip_Image_Caption_Large()
    image_path = "https://upload.wikimedia.org/wikipedia/commons/8/8f/Students_taking_computerized_exam.jpg"
    result = image_caption_model.caption_image(image_path, use_local_caption=True)
    assert result == "several people sitting at desks with computers in a classroom"

# Test the image caption API with wikipedia image
def test_blip_image_caption_api():
    image_caption_model = Blip_Image_Caption_Large()
    image_path = "https://upload.wikimedia.org/wikipedia/commons/8/8f/Students_taking_computerized_exam.jpg"
    result = image_caption_model.caption_image(image_path, use_local_caption=False)
    assert result == "several people sitting at desks with computers in a classroom"
    