"""Create the owner's phone link locally. Never write secrets in the repository."""
from pathlib import Path
import argparse
import json
import qrcode
import qrcode.image.svg

parser = argparse.ArgumentParser()
parser.add_argument('--session', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--player', action='store_true', help='Crea il QR per il player vocale del cellulare.')
args = parser.parse_args()
session = json.loads(args.session.read_text(encoding='utf-8-sig'))
route = '/sito/studio/player/' if args.player else '/sito/studio/'
prefix = 'player-qr' if args.player else 'tutor-qr'
link = session['Url'] + route + '#tutor-token=' + session['Token']
args.output.mkdir(parents=True, exist_ok=True)
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=7, border=4)
qr.add_data(link)
qr.make(fit=True)
qr.make_image(fill_color='#1a237e', back_color='white').save(args.output / (prefix + '.png'))
qr.make_image(image_factory=qrcode.image.svg.SvgPathImage).save(args.output / (prefix + '.svg'))
print('Codice QR creato nella cache privata del tutor.')
