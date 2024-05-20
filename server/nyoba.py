import re

tweet_text = "Sdgkn berita dri Kompas &amp IDN times"

# Menghapus "&amp;" dan melakukan pembersihan karakter tambahan
cleaned_text = re.sub(r'@[^\s]+', '', tweet_text)
cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
cleaned_text = re.sub(r'&amp;', '', cleaned_text)  # Menghapus "&amp;"
cleaned_text = re.sub(r'\bamp\b', '', cleaned_text)  # Menghapus "amp"
cleaned_text = re.sub(r'http\S+', '', cleaned_text)
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  
cleaned_text = cleaned_text.lower()

print(cleaned_text)