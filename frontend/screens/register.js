import * as React from "react";
import {
  View,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
  ScrollView,
} from "react-native";
import { TextInput, Button, Text } from "react-native-paper";
import axios from "axios";

const RegisterScreen = ({ navigation }) => {
  const [fullName, setFullName] = React.useState("");
  const [phone, setPhone] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [confirmPassword, setConfirmPassword] = React.useState("");

  const handleRegister = async () => {
    if (password !== confirmPassword) {
      console.error("Passwords do not match.");
      return;
    }

    try {
      const response = await axios.post(`${AUTH_BASE_URL}/register`, {
        full_name: fullName,
        phone_number: phone,
        email: email,
        password: password,
      });
      navigation.navigate("signIn");
    } catch (error) {
      console.error("Error registering user:", error);
      if (error.response) {
        console.log(error.response.data);
        console.log(error.response.status);
        console.log(error.response.headers);
      } else if (error.request) {
        console.log(error.request);
      } else {
        console.log("Error", error.message);
      }
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ScrollView contentContainerStyle={styles.innerContainer}>
        <Text style={styles.title}>Sign Up</Text>
        <TextInput
          label="Full Name"
          value={fullName}
          onChangeText={(text) => setFullName(text)}
          mode="flat"
          underlineColor="#fff"
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
        />
        <TextInput
          label="Phone No"
          value={phone}
          onChangeText={(text) => setPhone(text)}
          mode="flat"
          underlineColor="#fff"
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
          keyboardType="phone-pad"
        />
        <TextInput
          label="Email"
          value={email}
          onChangeText={(text) => setEmail(text)}
          mode="flat"
          underlineColor="#fff"
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
        />
        <TextInput
          label="Password"
          value={password}
          onChangeText={(text) => setPassword(text)}
          mode="flat"
          underlineColor="#fff"
          secureTextEntry
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
        />
        <TextInput
          label="Confirm Password"
          value={confirmPassword}
          onChangeText={(text) => setConfirmPassword(text)}
          mode="flat"
          underlineColor="#fff"
          secureTextEntry
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
        />
        <Button
          mode="elevated"
          onPress={handleRegister}
          style={styles.button}
          labelStyle={styles.buttonText}
        >
          REGISTER
        </Button>
        <TouchableOpacity
          style={styles.signIn}
          onPress={() => navigation.navigate("signIn")}
        >
          <Text style={styles.signInText}>
            Have an Account?{" "}
            <Text style={styles.signInButtonText}> Sign in</Text>
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fbf8f6",
  },
  innerContainer: {
    flexGrow: 1,
    justifyContent: "center",
    paddingHorizontal: "5%",
  },
  title: {
    fontSize: 30,
    fontWeight: "bold",
    marginBottom: 35,
    alignSelf: "center",
    color: "#000",
  },
  input: {
    backgroundColor: "#fff",
    marginBottom: 20,
    elevation: 4,
  },
  button: {
    marginTop: 40,
    borderRadius: 25,
    paddingVertical: 4,
    backgroundColor: "#765952",
  },
  buttonText: {
    color: "#FFF",
    fontWeight: "bold",
  },
  signIn: {
    marginTop: 25,
    alignItems: "center",
  },
  signInText: {
    color: "#000",
  },
  signInButtonText: {
    fontWeight: "bold",
    color: "#765952",
    fontSize: 16,
  },
});

export default RegisterScreen;
