import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt

# --- SOLUSI ERROR QWidget ---
# QApplication HARUS diinisialisasi sebelum widget apapun (termasuk QMessageBox) dibuat.
app = QApplication(sys.argv)

# Impor Backend
try:
    from citrus_predictor import CitrusPredictor, CLASS_NAMES
except ImportError:
    QMessageBox.critical(None, "Import Error", "Gagal mengimpor citrus_predictor.py. Pastikan file ada.")
    sys.exit(1)
except FileNotFoundError as e:
    # Tangani error FileNotFoundError spesifik dari CitrusPredictor._load_model
    QMessageBox.critical(None, "File Model Error", f"Model tidak ditemukan: {e}")
    sys.exit(1)
except Exception as e:
    QMessageBox.critical(None, "Model Error", f"Gagal memuat model PyTorch: {e}")
    sys.exit(1)
# --- End of SOLUSI ---

class CitrusClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.setWindowTitle("🍊 Citrus Disease Classifier (EfficientNetV2-L)")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        self.load_model_instance()

    def load_model_instance(self):
        """Memuat instance model (sudah diimpor di atas, ini hanya set atribut)."""
        try:
            self.status_label.setText("Model siap. Pilih gambar.")
            self.predictor = CitrusPredictor()
            self.predict_button.setEnabled(True)
        except Exception:
            # Kegagalan serius sudah ditangkap di blok try/except awal
            self.status_label.setText("ERROR: Model tidak dapat dijalankan.")
            self.predict_button.setEnabled(False)

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Panel Kiri (Kontrol dan Hasil)
        left_panel = QVBoxLayout()
        self.select_button = QPushButton("1. Pilih Gambar")
        self.select_button.clicked.connect(self.select_image)
        left_panel.addWidget(self.select_button)
        
        self.path_input = QLineEdit(readOnly=True)
        left_panel.addWidget(QLabel("Path Gambar:"))
        left_panel.addWidget(self.path_input)

        self.predict_button = QPushButton("2. Prediksi")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False) 
        left_panel.addWidget(self.predict_button)
        
        self.status_label = QLabel("Memulai...")
        self.status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_panel.addWidget(self.status_label)
        
        left_panel.addWidget(QLabel("--- Hasil ---"))
        self.result_label = QLabel("Hasil:")
        self.result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.result_label.setStyleSheet("color: darkgreen;")
        left_panel.addWidget(self.result_label)

        self.confidence_label = QLabel("Keyakinan:")
        left_panel.addWidget(self.confidence_label)
        
        self.details_label = QLabel("Detail Probabilitas:")
        self.details_label.setWordWrap(True)
        left_panel.addWidget(self.details_label)
        
        left_panel.addStretch(1)
        main_layout.addLayout(left_panel)

        # Panel Kanan (Tampilan Gambar)
        self.image_label = QLabel("Tampilan Gambar")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.image_label)
        right_panel.addStretch(1)
        
        main_layout.addLayout(right_panel)
        self.setLayout(main_layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 
            "Pilih Gambar Daun Jeruk", 
            "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.path_input.setText(file_name)
            self.display_image(file_name)
            self.result_label.setText("Hasil: Menunggu prediksi...")
            self.confidence_label.setText("Keyakinan:")
            self.details_label.setText("Detail Probabilitas:")
            self.status_label.setText("Gambar dimuat. Tekan 'Prediksi'.")

    def display_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText("Gagal memuat gambar.")
            return
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def run_prediction(self):
        image_path = self.path_input.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Perhatian", "Pilih file gambar yang valid.")
            return

        self.status_label.setText("Memprediksi...")
        QApplication.processEvents() # Paksa GUI update status

        # Panggil Prediksi
        predicted_class, confidence, details = self.predictor.predict(image_path)

        # Update GUI
        if predicted_class == "File Error":
            self.result_label.setText(f"Hasil: ERROR FILE")
            self.confidence_label.setText(f"Pesan: Gagal memproses gambar.")
            self.status_label.setText("Prediksi gagal.")
            return
            
        self.result_label.setText(f"Hasil: **{predicted_class}**")
        self.confidence_label.setText(f"Keyakinan: {confidence:.2f}%")
        
        detail_text = "Detail Probabilitas:\n"
        # Susun probabilitas agar mudah dibaca
        for k in CLASS_NAMES:
            detail_text += f"- {k}: {details.get(k, 'N/A')}\n"
        self.details_label.setText(detail_text)
        
        self.status_label.setText("Prediksi selesai.")

if __name__ == '__main__':
    ex = CitrusClassifierApp()
    ex.show()
    # Gunakan objek 'app' yang sudah dibuat di awal file
    sys.exit(app.exec())