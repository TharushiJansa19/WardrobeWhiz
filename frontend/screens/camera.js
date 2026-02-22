import React, { useState, useEffect } from "react";
import {
  View,
  StyleSheet,
  Text,
  Image,
  TouchableOpacity,
} from "react-native";
import * as MediaLibrary from "expo-media-library";
import * as ImagePicker from "expo-image-picker";
import { IconButton, Button, ActivityIndicator } from "react-native-paper";
import CustomModal from "./components/custom-modal";
import { BASE_URL } from "../lib/url";

const CameraScreen = ({ navigation }) => {
  const [hasPermission, setHasPermission] = useState(null);
  const [image, setImage] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [showImageOptions, setShowImageOptions] = useState(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    (async () => {
      const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
      const mediaLibraryStatus = await MediaLibrary.requestPermissionsAsync();
      setHasPermission(
        cameraStatus.status === "granted" &&
          mediaLibraryStatus.status === "granted"
      );
    })();
  }, []);

  const handleCheckSimilarItems = () => {
    if (image) {
      navigation.navigate("similar-items-generator", { imageUrl: image });
    } else {
      console.log("No image selected.");
    }
  };

  const handleSearchForMatchingItems = () => {
    if (image) {
      navigation.navigate("matching-items-generator", { imageUrl: image });
    } else {
      console.log("No image selected.");
    }
  };

  const handleAddToMyCloset = async () => {
    if (image) {
      setUploading(true);
      try {
        let formData = new FormData();
        const uniqueName = `upload_${new Date().getTime()}.jpg`;

        formData.append("image", {
          uri: image,
          type: "image/jpeg",
          name: uniqueName,
        });

        const response = await fetch(
          `${BASE_URL}/classify`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error("Failed to upload image");
        }

        const data = await response.json();
        console.log("Image sent to backend:", data);
      } catch (error) {
        console.error("Error sending image to backend:", error);
      } finally {
        setUploading(false);
      }
    }
    navigation.navigate("closet");
  };

  const handleImageResult = async (result) => {
    if (!result.cancelled) {
      console.log("URI before saving:", result.assets[0].uri);
      setImage(result.assets[0].uri);
      setShowImageOptions(true);
    }
  };

  const takePicture = async () => {
    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [3, 4],
      quality: 0.5,
    });
    await handleImageResult(result);
    setShowModal(false);
  };

  const pickImageAndUpload = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [3, 4],
      quality: 0.5,
    });
    await handleImageResult(result);
    setShowModal(false);
  };

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      <View style={styles.cameraContainer}>
        {!image ? (
          <Text style={styles.placeholderText}>No Image Selected</Text>
        ) : (
          <Image source={{ uri: image }} style={styles.imageContainer} />
        )}
      </View>
      {uploading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size={60} animating={true} color="#765952" />
        </View>
      ) : (
        showImageOptions && (
          <View style={styles.buttonContainer}>
            <Button
              icon="plus"
              mode="elevated"
              buttonColor="#765952"
              style={{ borderRadius: 0, marginBottom: 20 }}
              uppercase={true}
              labelStyle={{ fontSize: 15 }}
              textColor="#fff"
              onPress={() => {
                handleAddToMyCloset();
                // console.log("Added to your closet...");
              }}
            >
              Add to my closet
            </Button>
            <Button
              icon="magnify"
              mode="elevated"
              buttonColor="#765952"
              style={{ borderRadius: 0, marginBottom: 20 }}
              uppercase={true}
              labelStyle={{ fontSize: 15 }}
              textColor="#fff"
              onPress={() => {
                handleCheckSimilarItems();
              }}
            >
              Check for Similar Items
            </Button>
            <Button
              mode="elevated"
              buttonColor="#765952"
              style={{ borderRadius: 0, marginBottom: 20 }}
              uppercase={true}
              labelStyle={{ fontSize: 15 }}
              textColor="#fff"
              onPress={() => {
                handleSearchForMatchingItems();
              }}
            >
              Search for Matching Items
            </Button>
            <Button
              icon="camera"
              mode="elevated"
              buttonColor="#fff"
              style={{
                borderRadius: 0,
                borderColor: "#765952",
                borderWidth: 0.5,
              }}
              uppercase={true}
              labelStyle={{ fontSize: 15 }}
              textColor="#000"
              onPress={() => {
                setShowImageOptions(false);
                setShowModal(true);
              }}
            >
              Retake Image
            </Button>
          </View>
        )
      )}
      {!showImageOptions && (
        <TouchableOpacity
          onPress={() => setShowModal(true)}
          style={styles.captureButton}
        >
          <IconButton icon="camera" size={50} />
        </TouchableOpacity>
      )}
      <CustomModal
        visible={showModal}
        onClose={() => setShowModal(false)}
        onTakePicture={takePicture}
        onPickImageAndUpload={pickImageAndUpload}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  cameraContainer: {
    flex: 4,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#fff",
  },
  buttonContainer: {
    padding: 10,
    marginBottom: 20,
    marginHorizontal: "5%",
    width: "90%",
  },
  imageContainer: {
    width: "90%",
    height: "70%",
    alignSelf: "center",
    marginTop: "auto",
    marginBottom: "auto",
    resizeMode: "contain",
  },
  placeholderText: {
    fontSize: 18,
    color: "#999",
  },
  captureButton: {
    margin: 18,
    padding: 4,
    alignSelf: "center",
    borderRadius: 50,
    backgroundColor: "#F8F0E3",
    elevation: 2,
  },
  loadingContainer: {
    padding: 10,
    alignItems: "center",
    marginBottom: 10,
  },
});

export default CameraScreen;
