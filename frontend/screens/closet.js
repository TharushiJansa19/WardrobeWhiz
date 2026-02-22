import React, { useEffect, useState } from "react";
import {
  SafeAreaView,
  ScrollView,
  StyleSheet,
  View,
  TouchableOpacity,
} from "react-native";
import { Appbar } from "react-native-paper";
import SegmentedControlTab from "react-native-segmented-control-tab";
import axios from "axios";
import renderContent from "./components/render-content";
import renderOptions from "./components/render-options";
import { BASE_URL } from "../lib/url";

const MIN_ITEMS_CONTAINER_HEIGHT = "70%";

const MyClosetScreen = ({ navigation }) => {
  const [categories, setCategories] = useState([]);
  const [colors, setColors] = useState([]);
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState(0);
  const [items, setItems] = useState([]);
  const [currentCategory, setCurrentCategory] = useState(null);
  const [currentColor, setCurrentColor] = useState(null);
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

  const fetchItems = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${BASE_URL}/`, {
        params: {
          page: 1,
          per_page: 10,
        },
        
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.status === 200) {
        const imageIds = response.data.images;
        const imageUrlPromises = imageIds.map(getImageUrlById);
        const imageUrls = await Promise.all(imageUrlPromises);

        const validUrls = imageUrls.filter((url) => url);
        setItems(validUrls);
      } else {
        throw new Error(
          `Fetching items failed with status code: ${response.status}`
        );
      }
    } catch (error) {
      console.error("Error fetching items:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch categories from the backend
  const fetchCategories = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(
        `${BASE_URL}/categories`,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.status !== 200) {
        throw new Error("Fetching categories failed");
      }

      const data = response.data;
      console.log({ data });
      setCategories(data.categories);
    } catch (error) {
      console.error("Error fetching categories:", error);
    } finally {
      setIsLoading(false);
    }
  };

  //Fetch colors from the backend
  const fetchColors = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(
        `${BASE_URL}/colors`,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.status !== 200) {
        throw new Error("Fetching colors failed");
      }

      const data = response.data;
      console.log({ data });
      setColors(data.colors);
    } catch (error) {
      console.error("Error fetching colors:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCategoryPress = async (category) => {
    setCurrentCategory(category);
    setIsLoading(true);
    try {
      const response = await axios.get(
        `${BASE_URL}/images_by_category`,
        {
          params: {
            category: category,
            page: 1,
            per_page: 10,
          },
        }
      );
      if (response.status === 200 && response.data && response.data.images) {
        const imageIds = response.data.images;
        const imageUrlPromises = imageIds.map(getImageUrlById);
        const imageUrls = await Promise.all(imageUrlPromises);
        const validUrls = imageUrls.filter((url) => url);
        setItems(validUrls);
        console.log("CategoryImages:", validUrls);
      } else {
        console.error("Failed to fetch category items:", response.status);
      }
    } catch (error) {
      console.error("Error fetching category items:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleColorPress = async (color) => {
    setCurrentColor(color);
    setIsLoading(true);
    try {
      const response = await axios.get(
        `${BASE_URL}/images_by_colors`,
        {
          params: {
            color: color,
            page: 1,
            per_page: 10,
          },
        }
      );
      if (response.status === 200 && response.data && response.data.images) {
        const imageIds = response.data.images;
        const imageUrlPromises = imageIds.map(getImageUrlById);
        const imageUrls = await Promise.all(imageUrlPromises);
        const validUrls = imageUrls.filter((url) => url);
        setItems(validUrls);
        console.log("ColorImages:", validUrls);
      } else {
        console.error("Failed to fetch color items:", response.status);
      }
    } catch (error) {
      console.error("Error fetching color items:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchItems();
    fetchCategories();
  }, []);

  const handleTabChange = async (index) => {
    setSelectedSegmentIndex(index);
    if (index === 0) {
      fetchCategories();
    } else if (index === 1) {
      fetchColors();
    }
  };

  const handleAddItems = () => {
    navigation.navigate("camera");
  };

  const content = renderContent(items, handleAddItems, isLoading);
  const options = renderOptions(
    selectedSegmentIndex,
    categories,
    colors,
    handleCategoryPress,
    handleColorPress
  );

  return (
    <SafeAreaView style={styles.container}>
      <Appbar.Header style={styles.appBar}>
        <Appbar.Content title="My Closet" titleStyle={styles.appBarTitle} />
        <View style={styles.actionButtonContainer}>
          <TouchableOpacity
            onPress={handleAddItems}
            style={styles.actionButton}
          >
            <Appbar.Action icon="plus" color={styles.appBarAction.color} />
          </TouchableOpacity>
        </View>
      </Appbar.Header>
      <View style={styles.segmentedControlContainer}>
        <SegmentedControlTab
          values={["Categories", "Colors"]}
          selectedIndex={selectedSegmentIndex}
          onTabPress={handleTabChange}
          tabsContainerStyle={styles.segmentedControl}
          tabStyle={styles.segmentedControlTab}
          activeTabStyle={styles.segmentedControlActiveTab}
          tabTextStyle={styles.segmentedControlTabText}
          activeTabTextStyle={styles.segmentedControlActiveTabText}
        />
      </View>

      <ScrollView
        horizontal={true}
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.categoryScrollContainer}
      >
        {options}
      </ScrollView>

      <ScrollView style={styles.cardScrollView}>
        <View style={styles.itemsContainer}>{content}</View>
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
  segmentedControlContainer: {
    paddingVertical: 15,
  },
  segmentedControl: {
    marginHorizontal: 20,
  },
  segmentedControlTab: {
    borderColor: "#4B371C",
    borderWidth: 1,
    paddingVertical: 10,
  },
  segmentedControlActiveTab: {
    backgroundColor: "#4B371C",
  },
  segmentedControlTabText: {
    color: "#4B371C",
    fontWeight: "bold",
  },
  segmentedControlActiveTabText: {
    color: "#F5F5F5",
  },
  categoryScrollContainer: {
    alignItems: "center",
    paddingVertical: 20,
  },

  cardScrollView: {
    minHeight: MIN_ITEMS_CONTAINER_HEIGHT,
  },
  itemsContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    padding: 10,
  },
});

export default MyClosetScreen;