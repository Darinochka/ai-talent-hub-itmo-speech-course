import torch
import torchaudio
import os
from models import Generator
from t2spec_converter import TextToSpecConverter

def generate_audio(text, model_path, output_dir, device='cuda'):
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    t2s = TextToSpecConverter()
    mel_spec = t2s.text2spec(text)
    mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0).to(device)
    mel_spec = mel_spec.transpose(1, 2)
    
    with torch.no_grad():
        audio = generator(mel_spec)
    
    audio = audio.squeeze(0)
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio[0:1, :]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{text[:30].replace(' ', '_')}.wav")
    torchaudio.save(
        output_path,
        audio.cpu(),
        sample_rate=22050
    )
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--test_file', type=str, default='test_sentences.txt', help='Path to test sentences file')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Directory to save generated audio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    
    args = parser.parse_args()
    
    with open(args.test_file, 'r') as f:
        test_sentences = [line.strip() for line in f.readlines()]
    
    for i, text in enumerate(test_sentences):
        print(f"Generating audio for sentence {i+1}/{len(test_sentences)}")
        output_path = generate_audio(text, args.model_path, args.output_dir, args.device)
        print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()