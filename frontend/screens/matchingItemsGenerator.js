import axios from "axios";
import React, { useState, useEffect } from "react";
import {
  SafeAreaView,
  ScrollView,
  StyleSheet,
  View,
  TouchableOpacity,
  Text,
  Image,
} from "react-native";
import {
  Appbar,
  Card,
  IconButton,
  Button,
  ActivityIndicator,
} from "react-native-paper";
import { BASE_URL } from "../lib/url";

const CARD_WIDTH = "47%";
const CARD_HEIGHT = 200;
const MIN_ITEMS_CONTAINER_HEIGHT = "70%";

const MatchingItemsGeneratingScreen = ({ route, navigation }) => {
  const { imageUrl } = route.params;
  const [matchinItems, setMatchingItems] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

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

  const fetchMatchingItems = async () => {
    setIsLoading(true);
    let formData = new FormData();
    formData.append("image", {
      uri: imageUrl,
      type: "image/jpeg",
      name: `upload_${Date.now()}.jpg`,
    });

    try {
      const response = await axios.post(
        `${BASE_URL}/find_matching`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (
        response.status === 200 &&
        response.data &&
        Array.isArray(response.data.data)
      ) {
        const imageUrlPromises = response.data.data.map(getImageUrlById);
        const imageUrls = await Promise.all(imageUrlPromises);
        const validUrls = imageUrls.filter((url) => url);
        setMatchingItems(validUrls);
      } else {
        console.error("Failed to fetch matching items:", response.status);
      }
    } catch (error) {
      console.error("Error fetching matching items:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMatchingItems();
  }, [imageUrl]);

  const handleAddItems = () => {
    navigation.navigate("camera");
  };

  const tryAnotherItem = () => {
    navigation.goBack();
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <View style={styles.centered}>
          <ActivityIndicator size={60} animating={true} color="#765952" />
        </View>
      );
    }
    if (matchinItems.length === 0) {
      return (
        <TouchableOpacity style={styles.watermarkContainer}>
          <Button
            icon=""
            mode="elevated"
            buttonColor="#fff"
            style={[styles.watermarkIcon, { marginBottom: 10 }]}
            uppercase={true}
            labelStyle={{ fontSize: 15 }}
            textColor="#000"
            onPress={handleAddItems}
          >
            Try Another Item
          </Button>
          <Text style={styles.watermarkText}>
            There's no pairing items in your closet to this item
          </Text>
        </TouchableOpacity>
      );
    } else {
      return matchinItems.map((item, index) => (
        <View key={index} style={styles.card}>
          <Image
            source={{ uri: item }}
            style={{ width: "100%", height: "100%" }}
            resizeMode="contain"
          />
        </View>
      ));
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <Appbar.Header style={styles.appBar}>
        <Appbar.Content
          title="Pairing Items You Have"
          titleStyle={styles.appBarTitle}
        />
        <View style={styles.actionButtonContainer}>
          <TouchableOpacity
            onPress={handleAddItems}
            style={styles.actionButton}
          >
            <Appbar.Action icon="plus" color={styles.appBarAction.color} />
          </TouchableOpacity>
        </View>
      </Appbar.Header>

      <View style={styles.contentContainer}>
        <Card style={styles.imageCard}>
          <Card.Cover source={{ uri: imageUrl }} style={styles.cardImage} />
        </Card>

        <View style={styles.tryAnotherContainer}>
          <Button
            icon="camera"
            mode="elevated"
            buttonColor="#765952"
            style={{ borderRadius: 0, marginBottom: 20 }}
            labelStyle={{ fontSize: 15 }}
            textColor="#fff"
            onPress={tryAnotherItem}
          >
            Try Another Item
          </Button>
        </View>
      </View>

      <ScrollView style={styles.cardScrollView}>
        <View style={styles.itemsContainer}>{renderContent()}</View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F5F5",
  },
  appBar: {
    backgroundColor: "#4B371C",
    height: 85,
    justifyContent: "center",
    alignItems: "center",
  },
  actionButtonContainer: {
    borderRadius: 15,
    overflow: "hidden",
  },
  actionButton: {
    backgroundColor: "#000",
  },

  appBarTitle: {
    color: "#F5F5F5",
    fontWeight: "bold",
  },
  appBarAction: {
    color: "#fff",
  },
  imageContainer: {
    alignItems: "center",
    marginVertical: 20,
  },
  uploadedImage: {
    width: "80%",
    height: 200,
    resizeMode: "cover",
  },
  uploadedItemText: {
    marginTop: 10,
    fontSize: 18,
    fontWeight: "bold",
    color: "#4B371C",
  },
  categoryScrollContainer: {
    alignItems: "center",
    paddingVertical: 20,
  },
  categoryIcon: {
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 10,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#F8F0E3",
    elevation: 4,
  },
  categoryText: {
    textAlign: "center",
    color: "#4A3728",
    fontWeight: "bold",
  },
  cardScrollView: {
    // minHeight: MIN_ITEMS_CONTAINER_HEIGHT,
    paddingTop: 15,
  },

  itemsContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    padding: 10,
  },
  card: {
    marginBottom: 10,
    elevation: 4,
    backgroundColor: "#F8F0E3",
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
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
    textAlign: "center",
  },

  contentContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginHorizontal: 10,
    marginTop: 20,
    marginBottom: 20,
  },
  imageCard: {
    width: "47%",
    height: CARD_HEIGHT,
  },
  cardImage: {
    height: "100%",
    resizeMode: "cover",
    elevation: 4,
    borderRadius: 0,
    borderWidth: 3,
    borderColor: "#000",
  },
  tryAnotherContainer: {
    width: "47%",
    height: CARD_HEIGHT,
    justifyContent: "center",
    padding: 10,
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 60,
  },
});

export default MatchingItemsGeneratingScreen;
