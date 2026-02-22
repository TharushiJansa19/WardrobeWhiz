import React, { useState } from "react";
import {
  StyleSheet,
  View,
  TextInput,
  Keyboard,
  Text,
  ScrollView,
  Image,
  SafeAreaView,
} from "react-native";
import { ActivityIndicator, Button } from "react-native-paper";
import axios from "axios";
import { BASE_URL } from "../lib/url";

const MIN_ITEMS_CONTAINER_HEIGHT = "70%";

const OutfitFinderScreen = () => {
  const [text, setText] = useState("");
  const [items, setItems] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const getImageUrlById = async (imageId) => {
    try {
      const response = await axios.get(
        `${BASE_URL}/get_image`,
        {
          params: { image_id: imageId },
        }
      );
      if (response.status === 200) {
        return response.request.responseURL;
      } else {
        console.error(
          `Fetching image by id failed, status code: ${response.status}`
        );
        return null;
      }
    } catch (error) {
      console.error(`Error fetching image by id: ${imageId}, error: `, error);
      return null;
    }
  };

  const handleEnterPress = async () => {
    Keyboard.dismiss();
    setIsLoading(true);
    setErrorMessage("");
    try {
      const response = await axios.get(
        `${BASE_URL}/find_similar_by_text`,
        {
          params: { text: text },
        }
      );
      if (response.status === 200 && response.data.data) {
        const imageIds = response.data.data;
        const imageUrlPromises = imageIds.map(getImageUrlById);
        const imageUrls = await Promise.all(imageUrlPromises);
        setItems(imageUrls.filter((url) => url !== null));
      } else {
        console.error("Failed to fetch similar items:", response.status);
      }
    } catch (error) {
      console.error("Error fetching similar items:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <View style={styles.centered}>
          <ActivityIndicator size={60} animating={true} color="#765952" />
        </View>
      );
    }

    if (items.length > 0) {
      return items.map((item, index) => (
        <View key={index} style={styles.card}>
          <Image
            source={{ uri: item }}
            style={styles.image}
            resizeMode="contain"
          />
        </View>
      ));
    }

    return (
      <View style={styles.centered}>
        <Text style={styles.watermarkText}>
          Enter text above to search for items.
        </Text>
      </View>
    );
  };
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.input}
          placeholder="Enter your style or item to search"
          placeholderTextColor="#9A7B4F"
          value={text}
          onChangeText={setText}
          onSubmitEditing={handleEnterPress}
        />
        <Button
          mode="elevated"
          onPress={handleEnterPress}
          uppercase={true}
          style={styles.button}
          labelStyle={styles.buttonLabel}
          disabled={isLoading}
        >
          {isLoading ? "Loading..." : "Find"}
        </Button>
      </View>
      {(items.length > 0 || isLoading) && (
        <ScrollView style={styles.cardScrollView}>
          <View style={styles.itemsContainer}>{renderContent()}</View>
        </ScrollView>
      )}
    </SafeAreaView>
  );
};

export default OutfitFinderScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#3F301D",
  },
  searchContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  input: {
    width: "100%",
    backgroundColor: "#FFF",
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
    fontSize: 16,
    color: "#3C280D",
    marginBottom: 20,
  },
  button: {
    padding: 4,
    paddingHorizontal: 20,
    backgroundColor: "#765952",
    borderRadius: 0,
  },
  buttonLabel: {
    color: "#FFF",
  },
  cardScrollView: {
    flex: 1,
    backgroundColor: "#FFF",
    minHeight: MIN_ITEMS_CONTAINER_HEIGHT,
  },
  itemsContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    padding: 10,
  },
  card: {
    width: "47%",
    aspectRatio: 1,
    marginBottom: 10,
    elevation: 4,
    backgroundColor: "#F8F0E3",
  },
  watermarkContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 60,
  },
  watermarkIcon: {
    backgroundColor: "#e0e0e0",
    borderRadius: 24,
  },
  watermarkText: {
    color: "#cccccc",
    fontSize: 20,
    marginTop: 10,
  },
  centered: {
    flex: 1,
    marginTop: 20,
    justifyContent: "center",
    alignItems: "center",
  },
  image: {
    width: "100%",
    height: "100%",
  },
});
