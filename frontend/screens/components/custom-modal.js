import React from 'react';
import { View, StyleSheet, Text, TouchableOpacity } from 'react-native';
import { Modal as PaperModal } from 'react-native-paper';

const CustomModal = ({ visible, onClose, onTakePicture, onPickImageAndUpload }) => {
  return (
    <PaperModal visible={visible} onDismiss={onClose} contentContainerStyle={styles.modalContainer}>
      <TouchableOpacity onPress={onTakePicture} style={styles.modalOption}>
        <Text style={styles.modalOptionText}>Take Picture</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={onPickImageAndUpload} style={styles.modalOption}>
        <Text style={styles.modalOptionText}>Upload</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={onClose} style={styles.modalOption}>
        <Text style={styles.modalOptionText}>Cancel</Text>
      </TouchableOpacity>
    </PaperModal>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: 'white',
    padding: 20,
    margin: 40,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalOption: {
    padding: 10,
    alignItems: 'center',
  },
  modalOptionText: {
    fontSize: 18,
    color: '#333',
  },
});

export default CustomModal;
