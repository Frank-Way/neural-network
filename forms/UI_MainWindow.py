# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        font = QtGui.QFont()
        font.setPointSize(15)
        MainWindow.setFont(font)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setObjectName("startButton")
        self.gridLayout.addWidget(self.startButton, 2, 0, 1, 1)
        self.exitButton = QtWidgets.QPushButton(self.centralwidget)
        self.exitButton.setObjectName("exitButton")
        self.gridLayout.addWidget(self.exitButton, 2, 2, 1, 1)
        self.exportButton = QtWidgets.QPushButton(self.centralwidget)
        self.exportButton.setObjectName("exportButton")
        self.gridLayout.addWidget(self.exportButton, 2, 1, 1, 1)
        self.progressLabel = QtWidgets.QLabel(self.centralwidget)
        self.progressLabel.setText("")
        self.progressLabel.setObjectName("progressLabel")
        self.gridLayout.addWidget(self.progressLabel, 2, 3, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.inputsGroupBox = QtWidgets.QGroupBox(self.tab_1)
        self.inputsGroupBox.setMaximumSize(QtCore.QSize(300, 16777215))
        self.inputsGroupBox.setObjectName("inputsGroupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.inputsGroupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.sampleSizeEdit = QtWidgets.QLineEdit(self.inputsGroupBox)
        self.sampleSizeEdit.setClearButtonEnabled(True)
        self.sampleSizeEdit.setObjectName("sampleSizeEdit")
        self.gridLayout_4.addWidget(self.sampleSizeEdit, 6, 3, 1, 1)
        self.functionTextEdit = QtWidgets.QTextEdit(self.inputsGroupBox)
        self.functionTextEdit.setObjectName("functionTextEdit")
        self.gridLayout_4.addWidget(self.functionTextEdit, 2, 0, 1, 4)
        self.extendEdit = QtWidgets.QLineEdit(self.inputsGroupBox)
        self.extendEdit.setClearButtonEnabled(True)
        self.extendEdit.setObjectName("extendEdit")
        self.gridLayout_4.addWidget(self.extendEdit, 8, 3, 1, 1)
        self.inputsMinMaxTable = QtWidgets.QTableWidget(self.inputsGroupBox)
        self.inputsMinMaxTable.setRowCount(0)
        self.inputsMinMaxTable.setColumnCount(2)
        self.inputsMinMaxTable.setObjectName("inputsMinMaxTable")
        self.gridLayout_4.addWidget(self.inputsMinMaxTable, 5, 0, 1, 4)
        self.testSizeLabel = QtWidgets.QLabel(self.inputsGroupBox)
        self.testSizeLabel.setObjectName("testSizeLabel")
        self.gridLayout_4.addWidget(self.testSizeLabel, 7, 0, 1, 2)
        self.inputsMinMaxLabel = QtWidgets.QLabel(self.inputsGroupBox)
        self.inputsMinMaxLabel.setObjectName("inputsMinMaxLabel")
        self.gridLayout_4.addWidget(self.inputsMinMaxLabel, 4, 0, 1, 4)
        self.sampleSizeLabel = QtWidgets.QLabel(self.inputsGroupBox)
        self.sampleSizeLabel.setObjectName("sampleSizeLabel")
        self.gridLayout_4.addWidget(self.sampleSizeLabel, 6, 0, 1, 3)
        self.inputsLabel = QtWidgets.QLabel(self.inputsGroupBox)
        self.inputsLabel.setObjectName("inputsLabel")
        self.gridLayout_4.addWidget(self.inputsLabel, 0, 0, 1, 2)
        self.functionLabel = QtWidgets.QLabel(self.inputsGroupBox)
        self.functionLabel.setObjectName("functionLabel")
        self.gridLayout_4.addWidget(self.functionLabel, 1, 0, 1, 1)
        self.testSizeEdit = QtWidgets.QLineEdit(self.inputsGroupBox)
        self.testSizeEdit.setClearButtonEnabled(True)
        self.testSizeEdit.setObjectName("testSizeEdit")
        self.gridLayout_4.addWidget(self.testSizeEdit, 7, 3, 1, 1)
        self.extendLabel = QtWidgets.QLabel(self.inputsGroupBox)
        self.extendLabel.setObjectName("extendLabel")
        self.gridLayout_4.addWidget(self.extendLabel, 8, 0, 1, 2)
        self.inputsSpinBox = QtWidgets.QSpinBox(self.inputsGroupBox)
        self.inputsSpinBox.setMinimum(1)
        self.inputsSpinBox.setObjectName("inputsSpinBox")
        self.gridLayout_4.addWidget(self.inputsSpinBox, 0, 2, 1, 2)
        self.validateFunctionButton = QtWidgets.QPushButton(self.inputsGroupBox)
        self.validateFunctionButton.setMaximumSize(QtCore.QSize(110, 16777215))
        self.validateFunctionButton.setObjectName("validateFunctionButton")
        self.gridLayout_4.addWidget(self.validateFunctionButton, 3, 0, 1, 2)
        self.plotFunctionButton = QtWidgets.QPushButton(self.inputsGroupBox)
        self.plotFunctionButton.setMaximumSize(QtCore.QSize(110, 16777215))
        self.plotFunctionButton.setObjectName("plotFunctionButton")
        self.gridLayout_4.addWidget(self.plotFunctionButton, 3, 2, 1, 2)
        self.gridLayout_3.addWidget(self.inputsGroupBox, 0, 0, 1, 1)
        self.trainGroupBox = QtWidgets.QGroupBox(self.tab_1)
        self.trainGroupBox.setMaximumSize(QtCore.QSize(275, 16777215))
        self.trainGroupBox.setObjectName("trainGroupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.trainGroupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lrStartLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.lrStartLabel.setObjectName("lrStartLabel")
        self.gridLayout_2.addWidget(self.lrStartLabel, 0, 0, 1, 1)
        self.lrEdit = QtWidgets.QLineEdit(self.trainGroupBox)
        self.lrEdit.setClearButtonEnabled(True)
        self.lrEdit.setObjectName("lrEdit")
        self.gridLayout_2.addWidget(self.lrEdit, 0, 1, 1, 1)
        self.lrFinalLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.lrFinalLabel.setObjectName("lrFinalLabel")
        self.gridLayout_2.addWidget(self.lrFinalLabel, 1, 0, 1, 1)
        self.lrFinalEdit = QtWidgets.QLineEdit(self.trainGroupBox)
        self.lrFinalEdit.setClearButtonEnabled(True)
        self.lrFinalEdit.setObjectName("lrFinalEdit")
        self.gridLayout_2.addWidget(self.lrFinalEdit, 1, 1, 1, 1)
        self.decayLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.decayLabel.setObjectName("decayLabel")
        self.gridLayout_2.addWidget(self.decayLabel, 2, 0, 1, 1)
        self.decayComboBox = QtWidgets.QComboBox(self.trainGroupBox)
        self.decayComboBox.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
        self.decayComboBox.setObjectName("decayComboBox")
        self.gridLayout_2.addWidget(self.decayComboBox, 2, 1, 1, 1)
        self.epochsLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.epochsLabel.setObjectName("epochsLabel")
        self.gridLayout_2.addWidget(self.epochsLabel, 3, 0, 1, 1)
        self.epochsEdit = QtWidgets.QLineEdit(self.trainGroupBox)
        self.epochsEdit.setClearButtonEnabled(True)
        self.epochsEdit.setObjectName("epochsEdit")
        self.gridLayout_2.addWidget(self.epochsEdit, 3, 1, 1, 1)
        self.queryLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.queryLabel.setObjectName("queryLabel")
        self.gridLayout_2.addWidget(self.queryLabel, 4, 0, 1, 1)
        self.queryEdit = QtWidgets.QLineEdit(self.trainGroupBox)
        self.queryEdit.setClearButtonEnabled(True)
        self.queryEdit.setObjectName("queryEdit")
        self.gridLayout_2.addWidget(self.queryEdit, 4, 1, 1, 1)
        self.batchSizeLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.batchSizeLabel.setObjectName("batchSizeLabel")
        self.gridLayout_2.addWidget(self.batchSizeLabel, 5, 0, 1, 1)
        self.batchSizeEdit = QtWidgets.QLineEdit(self.trainGroupBox)
        self.batchSizeEdit.setClearButtonEnabled(True)
        self.batchSizeEdit.setObjectName("batchSizeEdit")
        self.gridLayout_2.addWidget(self.batchSizeEdit, 5, 1, 1, 1)
        self.stoppingLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.stoppingLabel.setObjectName("stoppingLabel")
        self.gridLayout_2.addWidget(self.stoppingLabel, 6, 0, 1, 1)
        self.stoppingCheckBox = QtWidgets.QCheckBox(self.trainGroupBox)
        self.stoppingCheckBox.setText("")
        self.stoppingCheckBox.setChecked(True)
        self.stoppingCheckBox.setObjectName("stoppingCheckBox")
        self.gridLayout_2.addWidget(self.stoppingCheckBox, 6, 1, 1, 1)
        self.printLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.printLabel.setObjectName("printLabel")
        self.gridLayout_2.addWidget(self.printLabel, 7, 0, 1, 1)
        self.printCheckBox = QtWidgets.QCheckBox(self.trainGroupBox)
        self.printCheckBox.setText("")
        self.printCheckBox.setChecked(True)
        self.printCheckBox.setObjectName("printCheckBox")
        self.gridLayout_2.addWidget(self.printCheckBox, 7, 1, 1, 1)
        self.plotsLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.plotsLabel.setObjectName("plotsLabel")
        self.gridLayout_2.addWidget(self.plotsLabel, 8, 0, 1, 1)
        self.plotsCheckBox = QtWidgets.QCheckBox(self.trainGroupBox)
        self.plotsCheckBox.setText("")
        self.plotsCheckBox.setChecked(True)
        self.plotsCheckBox.setObjectName("plotsCheckBox")
        self.gridLayout_2.addWidget(self.plotsCheckBox, 8, 1, 1, 1)
        self.lossLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.lossLabel.setObjectName("lossLabel")
        self.gridLayout_2.addWidget(self.lossLabel, 9, 0, 1, 1)
        self.lossComboBox = QtWidgets.QComboBox(self.trainGroupBox)
        self.lossComboBox.setObjectName("lossComboBox")
        self.gridLayout_2.addWidget(self.lossComboBox, 9, 1, 1, 1)
        self.optimizerLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.optimizerLabel.setObjectName("optimizerLabel")
        self.gridLayout_2.addWidget(self.optimizerLabel, 10, 0, 1, 1)
        self.optimizerComboBox = QtWidgets.QComboBox(self.trainGroupBox)
        self.optimizerComboBox.setObjectName("optimizerComboBox")
        self.gridLayout_2.addWidget(self.optimizerComboBox, 10, 1, 1, 1)
        self.momentumLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.momentumLabel.setObjectName("momentumLabel")
        self.gridLayout_2.addWidget(self.momentumLabel, 11, 0, 1, 1)
        self.momentumEdit = QtWidgets.QLineEdit(self.trainGroupBox)
        self.momentumEdit.setClearButtonEnabled(True)
        self.momentumEdit.setObjectName("momentumEdit")
        self.gridLayout_2.addWidget(self.momentumEdit, 11, 1, 1, 1)
        self.restartsLabel = QtWidgets.QLabel(self.trainGroupBox)
        self.restartsLabel.setObjectName("restartsLabel")
        self.gridLayout_2.addWidget(self.restartsLabel, 12, 0, 1, 1)
        self.restartsSpinBox = QtWidgets.QSpinBox(self.trainGroupBox)
        self.restartsSpinBox.setMinimum(1)
        self.restartsSpinBox.setObjectName("restartsSpinBox")
        self.gridLayout_2.addWidget(self.restartsSpinBox, 12, 1, 1, 1)
        self.gridLayout_3.addWidget(self.trainGroupBox, 0, 1, 1, 1)
        self.layersGroupBox = QtWidgets.QGroupBox(self.tab_1)
        self.layersGroupBox.setObjectName("layersGroupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.layersGroupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.layersLabel = QtWidgets.QLabel(self.layersGroupBox)
        self.layersLabel.setObjectName("layersLabel")
        self.gridLayout_6.addWidget(self.layersLabel, 0, 0, 1, 1)
        self.layersSpinBox = QtWidgets.QSpinBox(self.layersGroupBox)
        self.layersSpinBox.setMinimum(1)
        self.layersSpinBox.setObjectName("layersSpinBox")
        self.gridLayout_6.addWidget(self.layersSpinBox, 0, 1, 1, 1)
        self.layersTable = QtWidgets.QTableWidget(self.layersGroupBox)
        self.layersTable.setAlternatingRowColors(False)
        self.layersTable.setColumnCount(3)
        self.layersTable.setObjectName("layersTable")
        self.layersTable.setRowCount(0)
        self.layersTable.horizontalHeader().setDefaultSectionSize(100)
        self.gridLayout_6.addWidget(self.layersTable, 1, 0, 1, 2)
        self.gridLayout_3.addWidget(self.layersGroupBox, 0, 2, 1, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.outputEdit = QtWidgets.QTextEdit(self.tab_2)
        self.outputEdit.setReadOnly(True)
        self.outputEdit.setObjectName("outputEdit")
        self.gridLayout_7.addWidget(self.outputEdit, 0, 0, 1, 1)
        self.clearOutputButton = QtWidgets.QPushButton(self.tab_2)
        self.clearOutputButton.setObjectName("clearOutputButton")
        self.gridLayout_7.addWidget(self.clearOutputButton, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.mdiArea = QtWidgets.QMdiArea(self.tab_3)
        self.mdiArea.setObjectName("mdiArea")
        self.gridLayout_5.addWidget(self.mdiArea, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1000, 29))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menuBar)
        self.tabsAction = QtWidgets.QAction(MainWindow)
        self.tabsAction.setObjectName("tabsAction")
        self.buttonsAction = QtWidgets.QAction(MainWindow)
        self.buttonsAction.setObjectName("buttonsAction")
        self.guideAction = QtWidgets.QAction(MainWindow)
        self.guideAction.setObjectName("guideAction")
        self.aboutAction = QtWidgets.QAction(MainWindow)
        self.aboutAction.setObjectName("aboutAction")
        self.settingsAction = QtWidgets.QAction(MainWindow)
        self.settingsAction.setObjectName("settingsAction")
        self.menu.addAction(self.tabsAction)
        self.menu.addAction(self.buttonsAction)
        self.menu.addAction(self.settingsAction)
        self.menu.addSeparator()
        self.menu.addAction(self.guideAction)
        self.menu.addAction(self.aboutAction)
        self.menuBar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "???????????????? ?????????????????? ?????????? ?????? ?????????????????????????? ???????????????????????????? ??????????????"))
        self.startButton.setText(_translate("MainWindow", "????????????"))
        self.exitButton.setText(_translate("MainWindow", "??????????"))
        self.exportButton.setText(_translate("MainWindow", "??????????????"))
        self.inputsGroupBox.setTitle(_translate("MainWindow", "??????????????"))
        self.sampleSizeEdit.setPlaceholderText(_translate("MainWindow", "128"))
        self.functionTextEdit.setPlaceholderText(_translate("MainWindow", "x1^2-sqrt(x)*arcsin(x/5)"))
        self.extendEdit.setPlaceholderText(_translate("MainWindow", "0.15"))
        self.testSizeLabel.setText(_translate("MainWindow", "???????????????? ??????????"))
        self.inputsMinMaxLabel.setText(_translate("MainWindow", "?????????????????????? ????????????"))
        self.sampleSizeLabel.setText(_translate("MainWindow", "???????????? ??????????????"))
        self.inputsLabel.setText(_translate("MainWindow", "??????-???? ????????????"))
        self.functionLabel.setText(_translate("MainWindow", "F(X1)="))
        self.testSizeEdit.setPlaceholderText(_translate("MainWindow", "0.3"))
        self.extendLabel.setText(_translate("MainWindow", "????????????????????"))
        self.validateFunctionButton.setText(_translate("MainWindow", "????????????????"))
        self.plotFunctionButton.setText(_translate("MainWindow", "????????????"))
        self.trainGroupBox.setTitle(_translate("MainWindow", "????????????????"))
        self.lrStartLabel.setText(_translate("MainWindow", "??????. ????????."))
        self.lrEdit.setPlaceholderText(_translate("MainWindow", "0.1"))
        self.lrFinalLabel.setText(_translate("MainWindow", "??????. ????????."))
        self.lrFinalEdit.setPlaceholderText(_translate("MainWindow", "0.001"))
        self.decayLabel.setText(_translate("MainWindow", "????????????????"))
        self.epochsLabel.setText(_translate("MainWindow", "????????"))
        self.epochsEdit.setPlaceholderText(_translate("MainWindow", "1000"))
        self.queryLabel.setText(_translate("MainWindow", "??????????????"))
        self.queryEdit.setPlaceholderText(_translate("MainWindow", "10"))
        self.batchSizeLabel.setText(_translate("MainWindow", "??????????"))
        self.batchSizeEdit.setPlaceholderText(_translate("MainWindow", "64"))
        self.stoppingLabel.setText(_translate("MainWindow", "??????????. ??????."))
        self.printLabel.setText(_translate("MainWindow", "????????. ??????."))
        self.plotsLabel.setText(_translate("MainWindow", "??????????????"))
        self.lossLabel.setText(_translate("MainWindow", "????????????"))
        self.optimizerLabel.setText(_translate("MainWindow", "??????????????."))
        self.momentumLabel.setText(_translate("MainWindow", "??????????????"))
        self.momentumEdit.setPlaceholderText(_translate("MainWindow", "0.8"))
        self.restartsLabel.setText(_translate("MainWindow", "??????????????."))
        self.layersGroupBox.setTitle(_translate("MainWindow", "????????"))
        self.layersLabel.setText(_translate("MainWindow", "???????????????????????? ??????????"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "??????????????????"))
        self.outputEdit.setPlaceholderText(_translate("MainWindow", "?????????? ?????????? ???????????????? ???????????????????? ?? ?????????????????????? ????????????????"))
        self.clearOutputButton.setText(_translate("MainWindow", "????????????????"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "??????????"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "??????????????"))
        self.menu.setTitle(_translate("MainWindow", "????????????"))
        self.tabsAction.setText(_translate("MainWindow", "?? ???????????????????? ??????????????"))
        self.buttonsAction.setText(_translate("MainWindow", "?? ???????????????????? ????????????"))
        self.guideAction.setText(_translate("MainWindow", "????????????????????"))
        self.aboutAction.setText(_translate("MainWindow", "?? ??????????????????"))
        self.settingsAction.setText(_translate("MainWindow", "?? ????????????????????"))

