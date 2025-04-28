// metro.config.js
// 配置 Metro 打包器，决定你的项目资源（JS、图片等）怎么被打包和加载。

const { 
    wrapWithReanimatedMetroConfig,
  } = require('react-native-reanimated/metro-config');
  
  const config = {
    // 这里可以加你自己的 Metro 配置，如果没有可以留空
  };
  
  module.exports = wrapWithReanimatedMetroConfig(config);
  