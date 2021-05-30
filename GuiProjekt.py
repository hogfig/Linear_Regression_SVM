import sys
import random
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn import svm
import statsmodels.api as sm
pd.options.mode.chained_assignment = None



class MyWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        
        self.setStyleSheet("background-color:Cornsilk;") #postavlja boju pozadine
        self.setWindowTitle("Linerana regresija i SVM") #postavlja naslov aplikacije

        #counter koji sluzi za provjeru koliko puta smo usli u odredenu funkciju
        self.counter_linearna_reg = 0
        self.counter_train_linear = 0
        self.counter_svm = 0
        self.select_count = 0

        #definira se timer za svaku stranicu kako bi se omogucilo automatsko azuriranje podataka
        self.timer_linear_r = QTimer(self)
        self.timer_linear_r.setInterval(1000)
        self.timer_linear_tr = QTimer(self)
        self.timer_linear_tr.setInterval(1000)
        self.timer_svm = QTimer(self)
        self.timer_svm.setInterval(1000)

        #definiraju se stranice 
        self.homePageWidget = QWidget()
        self.secondPageWidget =  QWidget()
        self.thirdPageWidget =  QWidget()
        self.fourthPageWidget = QWidget()
        self.SvrPageWidget = QWidget()

        #stranice se grupiraju  
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.addWidget(self.homePageWidget)
        self.stackedWidget.addWidget(self.secondPageWidget)
        self.stackedWidget.addWidget(self.thirdPageWidget)
        self.stackedWidget.addWidget(self.fourthPageWidget)
        self.stackedWidget.addWidget(self.SvrPageWidget)

        #postavlja se layout i na taj layout se dodaju grupirane stranice
        self.Vlayout = QVBoxLayout()
        self.Vlayout.addWidget(self.stackedWidget)
        self.setLayout(self.Vlayout)

        #zove se prva stranica
        self.home_page()
        
        
    def home_page(self):
        '''
        Prva stranica služi za odabir .csv datoteke iz koje će se učitati podaci na kojima će se vršiti
        linearna ili SVM regresija.
        '''
        self.stackedWidget.setCurrentIndex(0)
        
        self.button = QPushButton()
        pixmap = QPixmap("file_48.png")

        button_icon = QIcon(pixmap)
        self.button.setIcon(button_icon)
        self.button.setIconSize(QSize(48,48))
        self.button.setFixedSize(38,48)
     
        self.text = QLabel("Odaberi datoteku za daljnje procesiranje")
        self.text.setFont(QFont('Times New Roman', 20))
        self.text.setAlignment(Qt.AlignCenter)

        Vlayout = QVBoxLayout()
        
        Vlayout.addWidget(self.text)
        Vlayout.addWidget(self.button,alignment=Qt.AlignCenter)

        self.homePageWidget.setLayout(Vlayout)

        # Spajamo signal, tj. kada se klikne definirani botun zove se određena funkcija 
        self.button.clicked.connect(self.read_file)
            
    def read_file(self):
        '''
        Funkcija za odabir .csv datoteke. Adresa datoteke se sprema u varijablu te se iz te adrese
        učitavaju podaci i spremaju u varijablu tipa dataframe. Zatim se poziva funkcija page_2().
        '''
        self.imena_stupaca=[]
        fname = QFileDialog.getOpenFileName()
        if fname[0] != '':
            
            self.data = pd.read_csv (fname[0]) #napravljeno samo za csv !
                                                #tu nadodaj logiku ako zelis vise od samo cvs fileova
            for col in self.data.columns: 
                self.imena_stupaca.append(col) #spremaju se imena stupaca u posebnu  varijablu.
       
            self.page_2()

    def page_2(self):
        '''
        Ova funkcija prikazuje korisničko sučelje u kojem se odabiru žejeni prediktori, zavisna varijabla,
        raspon podataka na kojima će se vršiti regresija te tip same regresije.
        '''

        Vlayout = QVBoxLayout()
        self.lista_check_box=[]
        self.grupa_check_box_NaN = []
        button1 = QPushButton("Select All")
        button2 = QPushButton("Linearna regresija")
        button3 = QPushButton("SVM")
        self.dropdown_picker = QComboBox()
        
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimum(2)
        self.slider.setMaximum(len(self.data))
        self.slider.setValue(int(len(self.data)/2))
        
        text1 = QLabel("Odaberi zavisnu varijablu:")
        text2 = QLabel("Odaberi prediktore:")
        text3 = QLabel("Odaberi raspon podataka za treniranje:")
        self.slider_text = QLabel(str(int(len(self.data)/2)))
        
        Vlayout.addWidget(text1)
        
        # Za odabir zavisne varijable stavljaju se svi stupci koji nemaju NaN vrijednosti
        for naslov in self.imena_stupaca:
            if ~self.data[naslov].isnull().values.any() and self.data[naslov].dtype != object:
                self.dropdown_picker.addItem(naslov)  

        Vlayout.addWidget(self.dropdown_picker,alignment= Qt.AlignLeft)
        Vlayout.addSpacing(12)
        Vlayout.addWidget(text2)

        for naslov in self.imena_stupaca:
            # ako u stupcu postoje NaN vrijednosti ili ako je tipa object njegov checkbox se povezuje sa funkcijom CheckBox_NaN()
            if self.data[naslov].isnull().values.any() or self.data[naslov].dtype == object:
                checkbox = QCheckBox(naslov, self)              
                checkbox.stateChanged.connect(lambda: self.CheckBox_NaN()) 
                self.grupa_check_box_NaN.append(checkbox)
                self.lista_check_box.append(checkbox)
                Vlayout.addWidget(checkbox)              
            else:
                checkbox = QCheckBox(naslov, self)
                self.lista_check_box.append(checkbox)
                Vlayout.addWidget(checkbox)               

        Vlayout.addWidget(button1,alignment= Qt.AlignLeft)
        Vlayout.addWidget(text3,alignment= Qt.AlignRight)
        Vlayout.addWidget(self.slider,alignment= Qt.AlignRight)
        Vlayout.addWidget(self.slider_text,alignment= Qt.AlignRight)
        Vlayout.addSpacing(2)
        Vlayout.addWidget(button2,alignment= Qt.AlignCenter)
        Vlayout.addWidget(button3,alignment= Qt.AlignCenter)
         
        self.secondPageWidget.setLayout(Vlayout)
        
        self.stackedWidget.setCurrentIndex(1)

        #connecting signals
        button1.clicked.connect(self.Check_all)
        self.slider.valueChanged.connect(self.changedSliderValue)
        button2.clicked.connect(self.redirect)
        button3.clicked.connect(self.redirect3)
        
    def redirect(self):
        '''
        Ako prvi put radimo linearnu regresiju pozovi cijelu funkciju Linearna_regresija(),
        uostalom procesiraj selektirane podatke, stvori model i samo se prebaci na
        Widget na kojem se prikazuje linearna regresija.
        '''
        if self.counter_linearna_reg == 0:
            self.Linearna_regresija()
        else:
            self.proces_data()
            self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
            

    def redirect2(self):
        '''
        Ako prvi put kliknemo na botun za trenirati model pozovi cijelu funkciju train_linear(),
        uostalom se prebaci na Widget na kojem se prikazuju podaci istreniranog modela linearne regresije.
        '''
        if self.counter_train_linear == 0:
            self.train_linear()
        else:
            self.stackedWidget.setCurrentIndex(3)

    def redirect3(self):
        '''
        Ako prvi put radimo SVR pozovi cijelu funkciju SvmPage(),
        uostalom procesiraj selektirane podatke, stvori model i samo se prebaci na
        Widget na kojem se prikazuje SVR.
        '''
        if self.counter_svm == 0:
            self.SvmPage()
        else:
            self.proces_svm()
            self.stackedWidget.setCurrentIndex(4)
                

    def Linearna_regresija(self):
        '''
        Prikaz dobivenih rezultata linearne regresije.
        '''
        #counter koji sluzi za provjeru koliko puta smo usli u funkciju Linearna_regresija()
        self.counter_linearna_reg += 1
        
        layout_linear = QVBoxLayout()
        self.button_back = QPushButton("Natrag")
        self.button_train = QPushButton("Treniraj model")

        self.proces_data() #funkcija za procesiranje podataka i stvaranje modela linearnom regresijom

        self.text = str(self.est.summary())
        self.text_lin_reg = QLabel()
        self.text_lin_reg.setText(self.text)

        # Automatski pozovi funkciju refresh svake sekunde
        self.timer_linear_r.timeout.connect(self.refresh)
        self.timer_linear_r.start()
        
        
        layout_linear.addWidget(self.button_back,alignment= Qt.AlignLeft)
        layout_linear.addWidget(self.text_lin_reg)
        layout_linear.addWidget(self.button_train,alignment= Qt.AlignRight)
        self.thirdPageWidget.setLayout(layout_linear)
        
        self.stackedWidget.setCurrentIndex(2)

        #connecting signals
        self.button_back.clicked.connect(self.Back)
        self.button_train.clicked.connect(self.redirect2)
        

    def train_linear(self):
        '''
        Funkcija koja prikazuje sumu kvadrata svih rezidualnih odstupanja,
        te omogućava vizualiziranje podataka.
        '''
        #counter koji sluzi za provjeru koliko puta smo usli u funkciju
        self.counter_train_linear += 1
        
        button_back = QPushButton('Natrag')
        button_plot = QPushButton('Vizualiziraj podatke')
        
        layout_linear = QVBoxLayout()
        text = QLabel('Suma kvadrata svih rezidualnih odstupanja:')
        self.koeficijenti = QLabel()

        self.koeficijenti.setText(str(self.S))

        # Automatski pozovi funkciju refresh1 svake sekunde
        self.timer_linear_tr.timeout.connect(self.refresh1)
        self.timer_linear_tr.start()


        layout_linear.addWidget(button_back,alignment= Qt.AlignLeft)
        layout_linear.addSpacing(12)
        layout_linear.addWidget(text,alignment= Qt.AlignTop)
        layout_linear.addWidget(self.koeficijenti,alignment= Qt.AlignTop)
        layout_linear.addSpacing(5)
        layout_linear.addWidget(button_plot,alignment= Qt.AlignLeft)

        self.fourthPageWidget.setLayout(layout_linear)
        self.stackedWidget.setCurrentIndex(3)

        # connect signal
        button_back.clicked.connect(self.Back)
        button_plot.clicked.connect(self.Plot)

    def SvmPage(self):
        '''
        Prikaz dobivenih rezultata SVR
        '''
        #counter koji sluzi za provjeru koliko puta smo usli u funkciju Linearna_regresija()
        self.counter_svm += 1

        layout = QVBoxLayout()
        plot_button = QPushButton("Vizualiziraj podatke")
        back_button = QPushButton("Natrag")

        self.proces_svm() #Funkcija za procesiranje podataka i izradu modela sa SVR

        layout.addWidget(back_button,alignment= Qt.AlignLeft)
        layout.addSpacing(12)
        
        self.model_score_text =QLabel("R^2 :  "+ str(self.model_score))  
        layout.addWidget(self.model_score_text)
        text =QLabel("Koeficijenti smijera:")
        layout.addWidget(text)
        
        self.svm_label = QLabel(self.svm_text)

        self.svm_suma = QLabel('Suma kvadrata svih rezidualnih odstupanja: '+str(self.S)) #

        # Automatski pozovi funkciju refresh2 svake sekunde
        self.timer_svm.timeout.connect(self.refresh2)
        self.timer_svm.start()

        layout.addWidget(self.svm_label)
        layout.addWidget(self.svm_suma )
        layout.addWidget(plot_button,alignment= Qt.AlignLeft)
        
        self.SvrPageWidget.setLayout(layout)
        self.stackedWidget.setCurrentIndex(4)

        #connect signal
        plot_button.clicked.connect(self.Plot)
        back_button.clicked.connect(self.Back)

    def proces_svm(self):
        '''
        Procesiraju se selektirani podaci i stvara se model sa SVR.
        '''
        self.prediktori= []
        self.zavisna_varijabla = self.dropdown_picker.currentText()
        self.training_range = range(0,self.slider.value())
        self.test_range = range(self.slider.value(),len(self.data))
       
        #Naslovi svih chekiranih stupaca koji ne sadrze NaN vrijednosti
        for i in self.lista_check_box:
             if i.isChecked() and ~self.data[i.text()].isnull().values.any():
                 self.prediktori.append(i.text())

        self.test_data_prediktori = self.data[self.prediktori].iloc[self.test_range, : ]
        self.test_data_zavisnaV = self.data[self.zavisna_varijabla].iloc[self.test_range].to_numpy()

        y = self.data[self.zavisna_varijabla].iloc[self.training_range].to_numpy().ravel()
        X = self.data[self.prediktori].iloc[self.training_range, : ].to_numpy()

        regr = svm.SVR(kernel='linear')
        model = regr.fit(X, y)
        self.R_squared = model.score(X,y)
        self.estimacije = model.predict(self.test_data_prediktori)
        self.model_score = model.score(X,y)
        self.Svm_koeficijetni = model.coef_
        self.Svm_konstanta = model.intercept_

        e = np.array(self.test_data_zavisnaV[:]-self.estimacije[:])
        self.S = np.array(e[:] * e[:]).sum()

        self.svm_text=''
        for i in range(len(self.prediktori)):
            self.svm_text = self.svm_text + str(self.prediktori[i]) + ': ' + str(self.Svm_koeficijetni[0][i]) + '\n\n'

        self.svm_text = self.svm_text + 'Konstanta: '+str(self.Svm_konstanta[0])

        if ~self.timer_svm.isActive():
            self.timer_svm.start()
        
    def proces_data(self):
        '''
        Procesiraju se selektirani podaci i stvara se model linearne regresije.
        '''
        self.prediktori= []
        self.zavisna_varijabla = self.dropdown_picker.currentText()
        self.training_range = range(0,self.slider.value())
        self.test_range = range(self.slider.value(),len(self.data))
       
        #Naslovi svih chekiranih stupaca koji ne sadrze NaN vrijednosti
        for i in self.lista_check_box:
             if i.isChecked() and ~self.data[i.text()].isnull().values.any():
                 self.prediktori.append(i.text())

        self.test_data_prediktori = self.data[self.prediktori].iloc[self.test_range, : ]
        self.test_data_prediktori = sm.add_constant(self.test_data_prediktori)
        self.test_data_zavisnaV = self.data[self.zavisna_varijabla].iloc[self.test_range].to_numpy()
                 
        y = self.data[self.zavisna_varijabla].iloc[self.training_range]
        X = self.data[self.prediktori].iloc[self.training_range, : ]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X)
        self.est = model.fit()

        self.estimacije = self.est.predict(self.test_data_prediktori).to_numpy()
        e = np.array(self.test_data_zavisnaV[:]-self.estimacije[:])
        self.S = np.array(e[:] * e[:]).sum()
        
        if ~self.timer_linear_r.isActive():        
            self.timer_linear_r.start()
        if ~self.timer_linear_tr.isActive():
            self.timer_linear_tr.start()

        
    def changedSliderValue(self):
        '''
        Omogucava prikaz selektirane vrijednosti sa sliderom
        '''
        size = self.slider.value()
        self.slider_text.setText(str(size))
        
    def CheckBox_NaN(self):
        '''
        Provjerava se ako selektiran stupac sadrzi NaN vrijednosti, ako sadrzi
        onemogucava se selektiranje.
        '''
        for i in self.grupa_check_box_NaN:
            i.setChecked(False)
        
    def Check_all(self):
        '''
        Prvim klikom na botun selektiraju se svi stupci osim onih koji u sebi sadrze NaN vrijednosti.
        Ako se klikne drugi put, odselektiraju se svi stupci.
        '''
        if self.select_count % 2 == 0:
            for i in self.lista_check_box:
                if self.data[i.text()].isnull().values.any():
                    i.setChecked(False)
                else:
                    i.setChecked(True)
        else:
            for i in self.lista_check_box:
                i.setChecked(False)

        self.select_count += 1
        
        
    def refresh(self):
       '''
        Azurira vrijednosti dobivenih linearnom regresijom.  
       '''
       x= str(self.est.summary())
       self.text_lin_reg.setText(x)
       
          
    def refresh1(self):
       '''
        Azurira vrijednosti  
       '''
       self.koeficijenti.setText(str(self.S))

    def refresh2(self):
       '''
        Azurira vrijednosti dobivene SVR.  
       '''
       self.svm_label.setText(self.svm_text)
       self.model_score_text.setText("R^2 :  "+ str(self.model_score))
       self.svm_suma .setText('Suma kvadrata svih rezidualnih odstupanja: '+str(self.S))
    
       
    def Back(self):
        '''
        Sluzi za povratak na predhodnu stranicu.
        '''
        if self.timer_linear_r.isActive():
            self.timer_linear_r.stop()
        if self.timer_linear_tr.isActive():
            self.timer_linear_tr.stop()
        if self.timer_svm.isActive():
            self.timer_svm.stop()
        
        if self.stackedWidget.currentIndex() == 4:
            self.stackedWidget.setCurrentIndex(1)
        else:    
            self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()-1)
        

    def Plot(self):
        '''
        Vizualizacija dobivenih rezultata.
        '''
        plt.figure()
        plt.plot(self.test_data_zavisnaV,label='Originalni podaci')
        plt.plot(self.estimacije,label='Estimirani podaci')
        plt.legend(loc='best')
        plt.figure()
        plt.scatter(self.test_data_zavisnaV,self.test_data_zavisnaV,label='Originalni podaci')
        plt.scatter(self.test_data_zavisnaV,self.estimacije,label='Estimirani podaci')
        plt.legend(loc='best')
        plt.show()

        
  

if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = MyWidget()
    widget.resize(350, 400)
    widget.show()

    sys.exit(app.exec_())
