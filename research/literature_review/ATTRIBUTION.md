# External Attribution and Legal Compliance

This file documents external sources, APIs, and libraries used in the literature review process, along with their terms of use and attribution requirements.

## External Data Sources

### arXiv API
- **Source**: https://arxiv.org/
- **API Documentation**: https://info.arxiv.org/help/api/index.html
- **Terms of Use**: https://arxiv.org/docs/tos/
- **License**: All arXiv content is available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license
- **Attribution Required**: Yes
- **Rate Limits**: 
  - No more than one request every 3 seconds per IP address
  - Scrapers should respect robots.txt
- **Data Usage**: 
  - arXiv papers are open access and can be used for research purposes
  - Proper attribution must be maintained when using paper metadata
  - Redistribution of paper content should include proper citation

### Python Libraries

#### arxiv
- **Repository**: https://github.com/lukasschwab/arxiv.py
- **License**: MIT License
- **Purpose**: Python wrapper for the arXiv API
- **Attribution**: MIT License allows free use, modification, and distribution with attribution

#### requests
- **Repository**: https://github.com/psf/requests
- **License**: Apache License 2.0
- **Purpose**: HTTP library for making API requests
- **Attribution**: Apache 2.0 License allows free use, modification, and distribution

## Citation Guidelines

When using data scraped from arXiv in publications or derivative works:

1. **Always cite the original paper** using its DOI or arXiv ID
2. **Include arXiv attribution** when aggregating or redistributing paper metadata
3. **Do not claim ownership** of scraped paper content
4. **Respect copyright**: Some papers may have been published in journals with different licensing terms

## Compliance Summary

- ✅ arXiv content is CC BY 4.0 licensed - free to use with attribution
- ✅ Used libraries are permissively licensed (MIT, Apache 2.0)
- ✅ Rate limiting is implemented in scrapers (0.3s delay between requests)
- ✅ Proper attribution is maintained in scraped metadata
- ⚠️ Users should verify individual paper licenses before reuse
- ⚠️ Check if papers have been published in journals with different licensing

## Contact

For questions about attribution or licensing, please refer to:
- arXiv Terms of Service: https://arxiv.org/docs/tos/
- arXiv API Help: https://info.arxiv.org/help/api/index.html

## Version History

- 2025-10-15: Initial attribution file created
