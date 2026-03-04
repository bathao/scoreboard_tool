import ollama

class OllamaVisionClient:
    def __init__(
        self,
        model_name="llama3.2-vision",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)

    def predict_winner(self, image_path: str) -> str:
        """
        Send image to Ollama and ask who won the rally.
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()

        prompt = (
            "This is a sequence of frames from a table tennis match. "
            "Analyze the last frame where the ball is dead. "
            "Based on player movement and ball position, who won the point? "
            "Respond with only one word: 'player_a', 'player_b', or 'unknown'."
        )

        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_data]
            }],
            options={
                "temperature": self.temperature
            }
        )

        result = response['message']['content'].lower().strip()
        
        # Simple cleanup
        if 'player_a' in result: return 'player_a'
        if 'player_b' in result: return 'player_b'
        return 'unknown'
