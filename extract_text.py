import sys
import re
from bs4 import BeautifulSoup

def convert_hocr_to_djvu(hocr_file, page_width, page_height):
    with open(hocr_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Iniciamos con la etiqueta de página usando las dimensiones reales
    output = f'(page 0 0 {page_width} {page_height}\n'
    
    for line in soup.find_all(class_='ocr_line'):
        title = line.get('title', '')
        bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
        if bbox_match:
            # Tesseract: x_min, y_min, x_max, y_max (desde arriba)
            x0, y0, x1, y1 = map(int, bbox_match.groups())
            
            # Conversión a coordenadas DjVu (desde abajo)
            djvu_y0 = page_height - y1
            djvu_y1 = page_height - y0
            
            text = line.get_text().replace('\n', ' ').replace('\r', ' ').strip()
            text = text.replace('"', '\\"')
            
            if text:
                output += f' (line {x0} {djvu_y0} {x1} {djvu_y1} "{text}")\n'
    
    output += ")" 
    return output

if __name__ == "__main__":
    # Uso: python hocr2djvu.py archivo.hocr ancho alto
    hocr_input = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    
    print(convert_hocr_to_djvu(hocr_input, width, height))