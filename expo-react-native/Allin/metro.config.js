const { getDefaultConfig } = require('expo/metro-config');
const { 
  wrapWithReanimatedMetroConfig,
} = require('react-native-reanimated/metro-config');

const config = getDefaultConfig(__dirname);

// 添加资源处理配置
config.resolver = {
  ...config.resolver,
  assetExts: [...config.resolver.assetExts, 'png', 'jpg', 'jpeg', 'gif'],
};

module.exports = wrapWithReanimatedMetroConfig(config);
  