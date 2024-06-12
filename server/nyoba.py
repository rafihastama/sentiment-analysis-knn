# Dokumen ke-16
document_16 = "berita kompas mesin framing kuik kuik kon kontol ong lah filter berita keluar die"

# Hitung total kata dalam dokumen
total_words_16 = len(document_16.split())

# Hitung jumlah kemunculan kata "kompas" dalam dokumen
kompas_count = document_16.split().count("berita")

# Hitung nilai TF untuk kata "kompas"
tf_kompas = kompas_count / total_words_16

print("Nilai TF untuk kata 'kompas' dalam dokumen ke-16:", tf_kompas)