from phi3_mini_4k_instruct import Phi3_Mini_4k_Instruct

# Test the local Phi3_Mini_4k_Instruct Model with default values
def test_phi3_mini_4k_instruct_local():
    phi3_mini_4k_instruct = Phi3_Mini_4k_Instruct()
    messages = [
        {"role": "system", "content": "You are an image caption to song description converter with a deep understanding of Music and Art. You are given the caption of an image. Your task is to generate a textual description of a musical piece that fits the caption. The description should be detailed and vivid, and should include the genre, mood, instruments, tempo, and other relevant information about the music. You should also use your knowledge of art and visual aesthetics to create a musical piece that complements the image. Only output the description of the music, without any explanation or introduction. Be concise."},
        {"role": "user", "content": "several people sitting at desks with computers in a classroom"},
    ]
    generated_description = phi3_mini_4k_instruct.generate_text(messages, use_local_llm=True)
    assert isinstance(generated_description, str) and generated_description != ""

def test_phi3_mini_4k_instruct_api():
    phi3_mini_4k_instruct = Phi3_Mini_4k_Instruct()
    messages = [
        {"role": "system", "content": "You are an image caption to song description converter with a deep understanding of Music and Art. You are given the caption of an image. Your task is to generate a textual description of a musical piece that fits the caption. The description should be detailed and vivid, and should include the genre, mood, instruments, tempo, and other relevant information about the music. You should also use your knowledge of art and visual aesthetics to create a musical piece that complements the image. Only output the description of the music, without any explanation or introduction. Be concise."},
        {"role": "user", "content": "several people sitting at desks with computers in a classroom"},
    ]
    generated_description = phi3_mini_4k_instruct.generate_text(messages, use_local_llm=False)
    assert isinstance(generated_description, str) and generated_description != ""